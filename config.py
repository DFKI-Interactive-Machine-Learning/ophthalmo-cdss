# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import os
from os.path import join
from src.segmentation import SegmentationModel
from plotly import express as px

# Paths
ROOT = os.path.dirname(os.path.realpath(__file__))

# Model Paths, old
CODE = join(ROOT, "src")
PROGNOSIS = join(CODE, "forecasting")
PROGNOSIS_WEIGHTS = join(PROGNOSIS, "weights")
SEGMENTATION = join(CODE, "segmentation")
SEGMENTATION_WEIGHTS = join(SEGMENTATION, "pretrained_models")

# Data Paths
DATA = join(ROOT, "data")
DATABASE = join(DATA, "demo.sqlite")
ML_DATA = join(DATA, "ML")
MERGED_DATAFRAMES = join(ML_DATA, "merged_dataframes")
OCT_TEST_FOLDER = join(DATA, "AROI_patient1")
TEMP_DATA = join(DATA, "temp")
ICONS = join(ROOT, "icons")


class LoggingConfig:
    level = "INFO"
    timing = False


# Keys for the database
class DatabaseKeys:
    visual_acuity = "Distance visual acuity"
    iop = "Intraocular pressure finding"
    drug = "Drug or medicament"
    visit_date = "visdate"


IvomDrugs = {
    "Control": 0,
    "Aflibercept": 1,
    "Brolucizumab": 2,
    "Dexamethasone": 3,
    "Fluocinolone acetonide": 4,
    "Pegaptanib": 5,
    "Product containing bevacizumab": 6,
    "Ranibizumab": 7,
    "Triamcinolone": 8,
    "Other IVOM medication": 9,
}

EyeStructures = ["Choroid",
                 "Sclera",
                 "Vitreous",
                 "Optic disc",
                 "Lens",
                 "Cornea",
                 "Iris",
                 "Conjunctiva",
                 "Peripapillary",
                 "Eyelid",
                 "Macula",
                 "Fovea",
                 "Pigment epithelial",
                 "Nerve fiber",
                 "Internal limiting membrane",
                 "External limiting membrane",
                 "Pupil",
                 "Anterior chamber",
                 "Retina",
                 "Yes",
                 ]
EyeStructures_to_index = {k: i for i, k in enumerate(EyeStructures)}
EyeSymptomes = {
    "edema": ["edema", "DME"],
    "bleeding": ["bleeding", "hemorrhage"],
    "aneurysm": ["aneurysm"],
    "inflammation": ["inflammation", "ivitis"],
    "scar": ["scar", "fibrosis"],
    "cataract": ["cataract"],
    "ischemia": ["ischemia"],
    "atrophy": ["atrophy"],
    "astigmatism": ["astigmatism"],
    "myopia": ["myopia"],
    "laser_burn": ["coagulation burn"],
    "neovascularization": ["neovascularization", "angiomatous proliferation", "rubeosis iridis"],
    "exudate": ["exudate", "cotton wool spots"],
    "cyst": ["cyst"],
    "tumor": ["neoplasm"],
    "hypertrophy": ["thickening", "hypertrophy"],
    "detachment": ["detachment"],
    "drusen": ["drusen", "lipid deposits", "deposition", "lipid"],
    "hyaline_bodies": ["hyaline body"]
}
Qualifiers = ["new", "fine", "large", "striped", "marginal", "small",
              "diffuse", "confluent", "flat", "surrounding", "circular", "peripheral",
              "superior", "inferior", "temporal", "nasal", "quiet"
              ]

# Use Cases
UseCases = {"AMD": 0,
            "DR": 1,
            "AMD_prospektiv": 100,
            "DME_prospektiv": 101
            }


# This list contains the verbose names of the features in the database
feature_names = (['Use case',
                  DatabaseKeys.drug,
                  'Age',
                  'Gender',
                  'Finding of tobacco smoking behavior',
                  'v_fluids',
                  'n_fluids',
                  'v_ped',
                  'n_ped',
                  'v_drusen',
                  'n_drusen',
                  'ipl_thickness',
                  'opl_thickness',
                  'elm_thickness',
                  'rpe_thickness',
                  'bm_thickness',
                  'choroidea_thickness',
                  'total_thickness',
                  "Distance visual acuity",
                  "Intraocular pressure finding"]
                 + list(EyeSymptomes.keys()) +
                 ["BMI",
                  "Days since last visit",
                  "Days since last treatment",
                  "Days since first visit",
                  "Days since first treatment",
                  "Number of IVOMs"])
feature_names_verbose = [
    "Diagnosis",
    "Medication",
    "Age",
    "Gender",
    "Smoker",
    "Volume of fluids",
    "Number of fluids",
    "Volume of PED",
    "Number of PED",
    "Volume of drusen",
    "Number of drusen",
    "IPL thickness",
    "OPL thickness",
    "ELM thickness",
    "RPE thickness",
    "BM thickness"
    "Total Retinal thickness",
    "Distance visual acuity",
    "IOP",
    "Edema",
    "Bleeding",
    "Aneurysm",
    "Inflammation",
    "Scar",
    "Cataract",
    "Ischemia",
    "Atrophy",
    "Astygmatism",
    "Myopia",
    "Laser burn",
    "Neovascularization",
    "Exudates",
    "Cysts",
    "Tumor",
    "Hypertrophy",
    "Detachment",
    "Drusen",
    "Hyaline bodies",
    "BMI",
    "Days since last visit",
    "Days since last treatment",
    "Days since first visit",
    "Days since first treatment",
    "Number of IVOMs",
]
va_unit = "decimal"
unique_meds = list(IvomDrugs.keys())
med_color_map = {med: px.colors.qualitative.Light24[i] for i, med in enumerate(sorted(unique_meds))}
segmentation_model = SegmentationModel()
LAYERS_TO_STRING = segmentation_model.classes
STRING_TO_LAYERS = {v: k for k, v in LAYERS_TO_STRING.items()}
RGB_COLORS = [
    (0, 0, 0),  # 1 Black
    (0, 255, 0),  # 2 Green
    (0, 0, 255),  # 3 Blue
    (255, 255, 0),  # 4 Yellow
    (255, 0, 255),  # 5 Purple
    (0, 255, 255),  # 6 Cyan
    (255, 140, 0),  # 7 Orange
    (0, 255, 0),  # 8 Choroidea
    (255, 0, 0),  # 9 Red
    (125, 125, 255),  # 10 Fluid
    (125, 0, 125),
    (0, 125, 125)
]
LAYERS_TO_COLOR = {layer: RGB_COLORS[i] for i, layer in enumerate(LAYERS_TO_STRING.keys())}
COLORS_TO_LAYER = {v: k for k, v in LAYERS_TO_COLOR.items()}
print(LAYERS_TO_STRING)
print(LAYERS_TO_COLOR)
