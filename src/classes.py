"""Human-readable names for the 43 GTSRB classes (index = ClassId)."""

CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing >3.5t", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles",
    "Vehicles >3.5t prohibited", "No entry", "General caution", "Dangerous curve left",
    "Dangerous curve right", "Double curve", "Bumpy road", "Slippery road",
    "Road narrows right", "Road work", "Traffic signals", "Pedestrians",
    "Children crossing", "Bicycles crossing", "Beware ice/snow", "Wild animals crossing",
    "End all limits", "Turn right ahead", "Turn left ahead", "Ahead only",
    "Straight or right", "Straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End no passing", "End no passing >3.5t",
]

assert len(CLASS_NAMES) == 43
