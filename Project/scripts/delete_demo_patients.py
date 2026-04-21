#!/usr/bin/env python3
"""Delete demo patients from the database."""

import requests

BASE_URL = "http://localhost:8001"

# Patients to delete (the 6 demo cases)
DEMO_PATIENTS = [
    "James Thompson",
    "Sarah Johnson", 
    "Jennifer Brown",
    "Robert Martinez",
    "Ahmed Hassan",
    "William Anderson"
]

def main():
    # Login
    login_response = requests.post(
        f"{BASE_URL}/auth/login",
        json={"email": "admin@example.com", "password": "admin123"}
    )
    
    if login_response.status_code != 200:
        print(f"❌ Error de autenticación: {login_response.status_code}")
        print(login_response.text)
        return
    
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get all patients
    patients_response = requests.get(f"{BASE_URL}/patients", headers=headers)
    patients = patients_response.json()
    
    print("=" * 60)
    print("     ELIMINANDO PACIENTES DE DEMO")
    print("=" * 60)
    
    deleted = 0
    for patient in patients:
        name = patient.get("name", "")
        if name in DEMO_PATIENTS:
            patient_id = patient["id"]
            delete_response = requests.delete(
                f"{BASE_URL}/patients/{patient_id}",
                headers=headers
            )
            if delete_response.status_code == 200:
                print(f"✅ Eliminado: {name}")
                deleted += 1
            else:
                print(f"❌ Error eliminando: {name}")
    
    print("=" * 60)
    print(f"     Total eliminados: {deleted}/{len(DEMO_PATIENTS)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
