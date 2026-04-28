"""
HMM Hospital Routing — pure functions, no matplotlib, no plotting.
Ported from matali.py for use in the backend.
Same math as the published simulation. Numpy used only internally.

Public API:
    select_hospital(call, hospitals=None) -> dict
        returns {'selected': str, 'utilities': [...], 'all_ranked': [...]}

    EmergencyCall(call_id, lat, lng, injury_type, severity)
    Hospital(name, lat, lng, tier, beds, capabilities, state_vector?)
    DEFAULT_HOSPITALS  -> 5 Gadag-region hospitals (canonical)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

STATES = ["Available", "Saturated", "Diverting"]


# ─────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────
@dataclass
class Hospital:
    name: str
    lat: float
    lng: float
    tier: int
    beds: int
    capabilities: dict
    state_vector: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.15, 0.05]))
    a12_base: float = 0.0
    k: float = 0.0
    dispatch_count: int = 0
    short: str = ""           # short label used by frontend
    id: str = ""              # stable id used by frontend (PHC/DH/GIMS/KIMS/SDM)

    def __post_init__(self):
        tier_base = {1: 0.02, 2: 0.05, 3: 0.12}
        self.a12_base = tier_base.get(self.tier, 0.05)
        self.k = 1.0 / max(self.beds, 1)


@dataclass
class EmergencyCall:
    call_id: int
    lat: float
    lng: float
    injury_type: str          # 'head' | 'chest' | 'limb' | 'cardiac'
    severity: str             # 'red' | 'yellow' | 'green'


# ─────────────────────────────────────────────────────────
# Core math (verbatim from matali.py)
# ─────────────────────────────────────────────────────────
def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def estimate_eta(distance_km, avg_speed_kmh=40):
    return (distance_km / avg_speed_kmh) * 60


def compute_dynamic_a12(a12_base, k, delta):
    return a12_base + (1 - a12_base) * (1 - np.exp(-k * delta))


def build_transition_matrix(hospital: Hospital):
    delta = hospital.dispatch_count
    a12 = compute_dynamic_a12(hospital.a12_base, hospital.k, delta)
    a13 = 0.005
    a11 = max(1 - a12 - a13, 0.0)
    a21 = 0.02
    a23 = min(0.03 + 0.02 * delta, 0.3)
    a22 = max(1 - a21 - a23, 0.0)
    a31 = 0.005
    a32 = 0.03
    a33 = max(1 - a31 - a32, 0.0)
    A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    for i in range(3):
        A[i] /= A[i].sum()
    return A


def forward_project(state_vector, A, delta_t):
    A_power = np.linalg.matrix_power(A, max(int(delta_t), 1))
    projected = state_vector @ A_power
    projected = np.clip(projected, 0, 1)
    projected /= projected.sum()
    return projected


def capability_score(hospital: Hospital, injury_type: str) -> float:
    caps = hospital.capabilities
    if injury_type == "head":
        return (1.0 if caps.get("neurosurgery") and caps.get("ct_scan") and caps.get("blood_bank")
                else 0.4 if caps.get("general_surgery") and caps.get("blood_bank")
                else 0.1 if caps.get("general_surgery")
                else 0.0)
    if injury_type == "chest":
        return (1.0 if caps.get("thoracic_surgery") and caps.get("blood_bank") and caps.get("icu")
                else 0.5 if caps.get("general_surgery") and caps.get("icu")
                else 0.1)
    if injury_type == "limb":
        return (1.0 if caps.get("orthopedics") and caps.get("blood_bank")
                else 0.6 if caps.get("general_surgery")
                else 0.2)
    if injury_type == "cardiac":
        return (1.0 if caps.get("cath_lab") and caps.get("icu")
                else 0.4 if caps.get("icu")
                else 0.1)
    return 0.5


def severity_lambda(severity: str) -> float:
    return {"red": 0.04, "yellow": 0.015, "green": 0.005}[severity]


def compute_utility(hospital: Hospital, call: EmergencyCall) -> dict:
    dist = haversine_km(call.lat, call.lng, hospital.lat, hospital.lng)
    eta = estimate_eta(dist)
    A = build_transition_matrix(hospital)
    projected = forward_project(hospital.state_vector, A, int(eta))
    p_available = float(projected[0])
    C = capability_score(hospital, call.injury_type)
    lam = severity_lambda(call.severity)
    U = p_available * C - lam * eta
    return {
        "hospital_id": hospital.id,
        "hospital": hospital.name,
        "short": hospital.short,
        "tier": hospital.tier,
        "distance_km": round(float(dist), 1),
        "eta_min": round(float(eta), 1),
        "p_available": round(p_available, 4),
        "p_saturated": round(float(projected[1]), 4),
        "p_diverting": round(float(projected[2]), 4),
        "capability": round(float(C), 2),
        "lambda": lam,
        "utility": round(float(U), 4),
        "dispatch_count": hospital.dispatch_count,
    }


# ─────────────────────────────────────────────────────────
# Canonical hospital roster — Gadag region
# Coords from matali.py (the math source of truth).
# ─────────────────────────────────────────────────────────
def default_hospitals() -> List[Hospital]:
    return [
        Hospital(
            id="GIMS", short="GIMS", name="GIMS Medical College",
            lat=15.4307, lng=75.6370, tier=1, beds=150,
            capabilities={
                "neurosurgery": True, "general_surgery": True, "orthopedics": True,
                "blood_bank": True, "ct_scan": True, "icu": True,
                "thoracic_surgery": True, "cath_lab": False,
            },
            state_vector=np.array([0.75, 0.18, 0.07]),
        ),
        Hospital(
            id="SDM", short="SDM", name="SDM Medical Dharwad",
            lat=15.4589, lng=75.0078, tier=1, beds=200,
            capabilities={
                "neurosurgery": True, "general_surgery": True, "orthopedics": True,
                "blood_bank": True, "ct_scan": True, "icu": True,
                "thoracic_surgery": True, "cath_lab": True,
            },
            state_vector=np.array([0.70, 0.20, 0.10]),
        ),
        Hospital(
            id="DH", short="DH", name="District Hospital Gadag",
            lat=15.4180, lng=75.6290, tier=2, beds=60,
            capabilities={
                "neurosurgery": False, "general_surgery": True, "orthopedics": True,
                "blood_bank": True, "ct_scan": True, "icu": True,
                "thoracic_surgery": False, "cath_lab": False,
            },
            state_vector=np.array([0.80, 0.15, 0.05]),
        ),
        Hospital(
            id="PVT", short="PVT", name="Private Trauma Center",
            lat=15.4250, lng=75.6400, tier=2, beds=30,
            capabilities={
                "neurosurgery": False, "general_surgery": True, "orthopedics": True,
                "blood_bank": True, "ct_scan": False, "icu": True,
                "thoracic_surgery": False, "cath_lab": False,
            },
            state_vector=np.array([0.85, 0.10, 0.05]),
        ),
        Hospital(
            id="PHC", short="PHC", name="PHC Lakkundi",
            lat=15.3900, lng=75.7100, tier=3, beds=10,
            capabilities={
                "neurosurgery": False, "general_surgery": False, "orthopedics": False,
                "blood_bank": False, "ct_scan": False, "icu": False,
                "thoracic_surgery": False, "cath_lab": False,
            },
            state_vector=np.array([0.90, 0.08, 0.02]),
        ),
    ]


# ─────────────────────────────────────────────────────────
# Public selection API
# ─────────────────────────────────────────────────────────
def select_hospital(call: EmergencyCall,
                    hospitals: Optional[List[Hospital]] = None) -> dict:
    """
    Run HMM utility calculation across all hospitals.
    Returns:
        {
          'selected': {hospital_id, hospital, utility, ...},
          'rejected_in_order': [hospital_id, ...]   # worst→best, excludes winner
          'all_ranked':        [util_dict, ...]      # best→worst
          'second_best_gap':   float                 # margin of victory
          'low_confidence':    bool                  # winner_utility < threshold
        }
    """
    if hospitals is None:
        hospitals = default_hospitals()

    utilities = [compute_utility(h, call) for h in hospitals]
    ranked = sorted(utilities, key=lambda u: u["utility"], reverse=True)
    selected = ranked[0]

    # Rejection order = worst → second-best (so frontend can dim them in this order)
    rejected_worst_to_best = list(reversed(ranked[1:]))
    rejected_ids = [u["hospital_id"] for u in rejected_worst_to_best]

    # Confidence checks
    margin = ranked[0]["utility"] - ranked[1]["utility"] if len(ranked) > 1 else 1.0
    low_confidence = (selected["utility"] < 0.0) or (margin < 0.05)

    return {
        "selected": selected,
        "rejected_in_order": rejected_ids,
        "all_ranked": ranked,
        "margin": round(float(margin), 4),
        "low_confidence": low_confidence,
    }


def reason_text(util: dict, winner: dict) -> str:
    """Human-readable rejection reason for a utility row."""
    if util["capability"] < 0.3:
        return f"Capability {util['capability']:.2f} · cannot handle this injury class"
    if util["p_available"] < 0.4:
        return f"P(Avail|ETA={util['eta_min']:.0f}min) = {util['p_available']:.2f} · projected saturation"
    if util["eta_min"] > 60:
        return f"ETA {util['eta_min']:.0f}min · time penalty exceeds capability gain"
    # Generic
    return (f"U={util['utility']:+.3f} vs winner {winner['utility']:+.3f} "
            f"(C={util['capability']:.2f}, P={util['p_available']:.2f}, ETA={util['eta_min']:.0f}m)")
