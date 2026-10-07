"""Shared constants and helpers."""

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# One slot per (plane, fat-suppressed fluid-sensitive) combination seen in the data.
SLOTS = ["Sagittal_0", "Sagittal_1", "Coronal_0", "Coronal_1", "Axial_0", "Axial_1"]


def series_slot(row):
    return f"{row['Anatomical_Plane']}_{int(row['Fat_Suppression'])}"
