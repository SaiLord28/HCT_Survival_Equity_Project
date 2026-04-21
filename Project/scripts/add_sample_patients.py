#!/usr/bin/env python3
"""
Add Sample Patients Script
===========================
Creates 10 diverse patient cases in the database and generates predictions.

Usage:
    python add_sample_patients.py
"""

import requests
import json
from datetime import datetime

BACKEND_URL = "http://localhost:8001"
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "admin123"

# 10 Diverse patient cases with realistic clinical data
SAMPLE_PATIENTS = [
    {
        "name": "Sarah Johnson",
        "description": "Young AML, MSD, excellent prognosis",
        "expected_risk": "Low",
        "data": {
            "age_at_hct": 25, "year_hct": 2024, "race_group": "White", "ethnicity": "Not Hispanic",
            "donor_age": 28, "donor_related": "MSD", "sex_match": "Matched",
            "prim_disease_hct": "AML", "dri_score": "Low", "cyto_score": "Good", "mrd_hct": "Negative",
            "conditioning_intensity": "MAC", "graft_type": "PBSC", "gvhd_proph": "Tacrolimus/MTX", "tbi_status": "No TBI",
            "hla_match_a_high": 2, "hla_match_b_high": 2, "hla_match_c_high": 2, "hla_match_drb1_high": 2, "hla_high_res_8": 8,
            "karnofsky_score": 100, "comorbidity_score": 0
        }
    },
    {
        "name": "Michael Williams",
        "description": "Middle-aged MDS, MUD, moderate risk",
        "expected_risk": "Medium",
        "data": {
            "age_at_hct": 52, "year_hct": 2024, "race_group": "White", "ethnicity": "Not Hispanic",
            "donor_age": 38, "donor_related": "MUD", "sex_match": "Mismatched",
            "prim_disease_hct": "MDS", "dri_score": "Intermediate", "cyto_score": "Intermediate", "mrd_hct": "Positive",
            "conditioning_intensity": "RIC", "graft_type": "PBSC", "gvhd_proph": "Tacrolimus/MMF", "tbi_status": "TBI 200cGy",
            "hla_match_a_high": 2, "hla_match_b_high": 2, "hla_match_c_high": 1, "hla_match_drb1_high": 2, "hla_high_res_8": 7,
            "karnofsky_score": 80, "comorbidity_score": 2, "diabetes": "Y"
        }
    },
    {
        "name": "Ahmed Hassan",
        "description": "Elderly CML crisis, poor prognosis",
        "expected_risk": "High",
        "data": {
            "age_at_hct": 67, "year_hct": 2023, "race_group": "White", "ethnicity": "Not Hispanic",
            "donor_age": 45, "donor_related": "MUD", "sex_match": "Mismatched",
            "prim_disease_hct": "CML", "dri_score": "Very High", "cyto_score": "Poor", "mrd_hct": "Positive",
            "conditioning_intensity": "MAC", "graft_type": "PBSC", "gvhd_proph": "Tacrolimus/MTX", "tbi_status": "TBI 1200cGy",
            "hla_match_a_high": 2, "hla_match_b_high": 1, "hla_match_c_high": 1, "hla_match_drb1_high": 2, "hla_high_res_8": 6,
            "karnofsky_score": 60, "comorbidity_score": 5, "cardiac": "Y", "diabetes": "Y", "pulm_moderate": "Y"
        }
    },
    {
        "name": "Emily Chen",
        "description": "Pediatric ALL, haploidentical",
        "expected_risk": "Medium",
        "data": {
            "age_at_hct": 10, "year_hct": 2024, "race_group": "Asian", "ethnicity": "Not Hispanic",
            "donor_age": 38, "donor_related": "Haploidentical", "sex_match": "Matched",
            "prim_disease_hct": "ALL", "dri_score": "Intermediate", "cyto_score": "Intermediate", "mrd_hct": "Negative",
            "conditioning_intensity": "MAC", "graft_type": "BM", "gvhd_proph": "PTCy/Tacrolimus", "tbi_status": "TBI 1200cGy",
            "hla_match_a_high": 1, "hla_match_b_high": 1, "hla_match_c_high": 1, "hla_match_drb1_high": 2, "hla_high_res_8": 5,
            "karnofsky_score": 100, "comorbidity_score": 0
        }
    },
    {
        "name": "James Thompson",
        "description": "Autologous myeloma, good status",
        "expected_risk": "Low",
        "data": {
            "age_at_hct": 58, "year_hct": 2024, "race_group": "Black or African American", "ethnicity": "Not Hispanic",
            "donor_age": 58, "donor_related": "Autologous", "sex_match": "Matched",
            "prim_disease_hct": "MM", "dri_score": "Low", "cyto_score": "Good", "mrd_hct": "Negative",
            "conditioning_intensity": "MAC", "graft_type": "PBSC", "gvhd_proph": "None", "tbi_status": "No TBI",
            "hla_match_a_high": 2, "hla_match_b_high": 2, "hla_match_c_high": 2, "hla_match_drb1_high": 2, "hla_high_res_8": 8,
            "karnofsky_score": 90, "comorbidity_score": 1, "obesity": "Y"
        }
    },
    {
        "name": "Fatima Al-Rahman",
        "description": "Young ALL, sibling donor",
        "expected_risk": "Low",
        "data": {
            "age_at_hct": 22, "year_hct": 2024, "race_group": "White", "ethnicity": "Not Hispanic",
            "donor_age": 19, "donor_related": "MSD", "sex_match": "Matched",
            "prim_disease_hct": "ALL", "dri_score": "Low", "cyto_score": "Good", "mrd_hct": "Negative",
            "conditioning_intensity": "MAC", "graft_type": "PBSC", "gvhd_proph": "Tacrolimus/MTX", "tbi_status": "TBI 1200cGy",
            "hla_match_a_high": 2, "hla_match_b_high": 2, "hla_match_c_high": 2, "hla_match_drb1_high": 2, "hla_high_res_8": 8,
            "karnofsky_score": 100, "comorbidity_score": 0
        }
    },
    {
        "name": "Robert Martinez",
        "description": "Elderly lymphoma, comorbidities",
        "expected_risk": "High",
        "data": {
            "age_at_hct": 71, "year_hct": 2023, "race_group": "Hispanic/Latino", "ethnicity": "Hispanic or Latino",
            "donor_age": 50, "donor_related": "MUD", "sex_match": "Mismatched",
            "prim_disease_hct": "NHL", "dri_score": "High", "cyto_score": "Poor", "mrd_hct": "Positive",
            "conditioning_intensity": "RIC", "graft_type": "PBSC", "gvhd_proph": "Tacrolimus/MMF", "tbi_status": "TBI 200cGy",
            "hla_match_a_high": 2, "hla_match_b_high": 1, "hla_match_c_high": 2, "hla_match_drb1_high": 2, "hla_high_res_8": 7,
            "karnofsky_score": 50, "comorbidity_score": 6, "cardiac": "Y", "arrhythmia": "Y", "renal_issue": "Y", "pulm_severe": "Y"
        }
    },
    {
        "name": "David Kim",
        "description": "Middle-aged AML, good match",
        "expected_risk": "Medium",
        "data": {
            "age_at_hct": 48, "year_hct": 2024, "race_group": "Asian", "ethnicity": "Not Hispanic",
            "donor_age": 32, "donor_related": "MUD", "sex_match": "Matched",
            "prim_disease_hct": "AML", "dri_score": "Intermediate", "cyto_score": "Intermediate", "mrd_hct": "Negative",
            "conditioning_intensity": "MAC", "graft_type": "PBSC", "gvhd_proph": "Tacrolimus/MTX", "tbi_status": "No TBI",
            "hla_match_a_high": 2, "hla_match_b_high": 2, "hla_match_c_high": 2, "hla_match_drb1_high": 2, "hla_high_res_8": 8,
            "karnofsky_score": 90, "comorbidity_score": 1, "hepatic_mild": "Y"
        }
    },
    {
        "name": "Jennifer Brown",
        "description": "Cord blood transplant",
        "expected_risk": "Medium",
        "data": {
            "age_at_hct": 35, "year_hct": 2024, "race_group": "Black or African American", "ethnicity": "Not Hispanic",
            "donor_age": 0, "donor_related": "UCB", "sex_match": "Unknown",
            "prim_disease_hct": "AML", "dri_score": "Intermediate", "cyto_score": "Intermediate", "mrd_hct": "Negative",
            "conditioning_intensity": "RIC", "graft_type": "UCB", "gvhd_proph": "CSA/MMF", "tbi_status": "TBI 200cGy",
            "hla_match_a_high": 1, "hla_match_b_high": 1, "hla_match_c_high": 1, "hla_match_drb1_high": 1, "hla_high_res_8": 4,
            "karnofsky_score": 80, "comorbidity_score": 1
        }
    },
    {
        "name": "William Anderson",
        "description": "Very high risk - multiple factors",
        "expected_risk": "High",
        "data": {
            "age_at_hct": 65, "year_hct": 2023, "race_group": "White", "ethnicity": "Not Hispanic",
            "donor_age": 55, "donor_related": "MMUD", "sex_match": "Mismatched",
            "prim_disease_hct": "MDS", "dri_score": "Very High", "cyto_score": "Very Poor", "mrd_hct": "Positive",
            "conditioning_intensity": "MAC", "graft_type": "PBSC", "gvhd_proph": "Tacrolimus/MTX", "tbi_status": "TBI 1200cGy",
            "hla_match_a_high": 1, "hla_match_b_high": 1, "hla_match_c_high": 1, "hla_match_drb1_high": 1, "hla_high_res_8": 4,
            "karnofsky_score": 50, "comorbidity_score": 7, "cardiac": "Y", "arrhythmia": "Y", "diabetes": "Y", 
            "hepatic_mild": "Y", "pulm_severe": "Y", "renal_issue": "Y", "prior_tumor": "Y"
        }
    }
]


def login():
    """Login and get token."""
    response = requests.post(
        f"{BACKEND_URL}/auth/login",
        json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    )
    if response.status_code != 200:
        raise Exception(f"Login failed: {response.text}")
    return response.json()["access_token"]


def create_patient(token, patient_data):
    """Create a patient and return the ID."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{BACKEND_URL}/patients",
        json=patient_data,
        headers=headers
    )
    if response.status_code != 200:
        raise Exception(f"Failed to create patient: {response.text}")
    return response.json()


def create_prediction(token, patient_id):
    """Create a prediction for a patient."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{BACKEND_URL}/predictions",
        json={"patient_id": patient_id},
        headers=headers
    )
    if response.status_code != 200:
        raise Exception(f"Failed to create prediction: {response.text}")
    return response.json()


def main():
    print("\n" + "="*70)
    print("   HCT SURVIVAL PREDICTION - SAMPLE PATIENT CREATION")
    print("="*70)
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Backend: {BACKEND_URL}")
    
    # Login
    try:
        token = login()
        print("   Status: ✅ Authenticated\n")
    except Exception as e:
        print(f"   Status: ❌ Login failed - {e}\n")
        return
    
    print("-"*70)
    print(f"{'#':<3} {'Name':<22} {'Expected':<8} {'Actual':<8} {'Prob':<8} {'Match'}")
    print("-"*70)
    
    results = {"passed": 0, "failed": 0}
    
    for i, patient_info in enumerate(SAMPLE_PATIENTS, 1):
        try:
            # Prepare patient data
            patient_data = {"name": patient_info["name"]}
            patient_data.update(patient_info["data"])
            
            # Create patient
            patient = create_patient(token, patient_data)
            
            # Create prediction
            prediction = create_prediction(token, patient["id"])
            
            # Compare results
            actual = prediction["risk_category"]
            expected = patient_info["expected_risk"]
            prob = prediction["event_probability"] * 100
            match = "✅" if actual == expected else "⚠️"
            
            if actual == expected:
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            print(f"{i:<3} {patient_info['name']:<22} {expected:<8} {actual:<8} {prob:>5.1f}%  {match}")
            
        except Exception as e:
            print(f"{i:<3} {patient_info['name']:<22} ERROR: {str(e)[:30]}")
            results["failed"] += 1
    
    print("-"*70)
    print(f"\n   SUMMARY: {results['passed']} matched expected, {results['failed']} different")
    print("   Note: Different results may indicate model calibration differences")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
