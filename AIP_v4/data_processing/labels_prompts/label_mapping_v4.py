# tag categories: period, style, mood, subject, visual elements

PERIOD_TAGS = [
    "pre_renaissance",    # before 1400
    "renaissance_era",    # 1400–1600
    "baroque_era",        # 1600–1700
    "18th_century",       # 1700–1800
    "19th_century",       # 1800–1900
    "early_modern",       # 1900–1960
    "contemporary",       # 1960+
]

STYLE_TAGS = [
    "realistic", "impressionist", "abstract", "expressionist",
    "cubist", "baroque", "renaissance", "romantic",
    "minimalist", "pop_art", "ukiyo_e"
]

MOOD_TAGS = [
    "joyful", "melancholic", "dramatic", "peaceful",
    "dark", "mysterious", "tense", "spiritual", "energetic", "serene"
]

ELEMENT_TAGS = [
    "warm_tones", "cool_tones", "vivid_colors", "muted_tones",
    "high_contrast", "soft_light", "dark_shadows", "geometric_shapes", "rich_texture"
]

SUBJECT_TAGS = [
    "portrait", "landscape", "still_life", "religious_mythological",
    "historical_battle", "everyday_life", "sea_water", "urban_city",
    "animals", "nude", "abstract_nonrepresentational"
]

ALL_TAGS = PERIOD_TAGS + STYLE_TAGS + MOOD_TAGS + ELEMENT_TAGS + SUBJECT_TAGS

GENRE_TO_PERIOD = {
    "Abstract_Expressionism":    "early_modern",
    "Action_painting":           "early_modern",
    "Analytical_Cubism":         "early_modern",
    "Art_Nouveau_Modern":        "19th_century",
    "Baroque":                   "baroque_era",
    "Color_Field_Painting":      "contemporary",
    "Contemporary_Realism":      "contemporary",
    "Cubism":                    "early_modern",
    "Early_Renaissance":         "renaissance_era",
    "Expressionism":             "early_modern",
    "Fauvism":                   "early_modern",
    "High_Renaissance":          "renaissance_era",
    "Impressionism":             "19th_century",
    "Mannerism_Late_Renaissance": "renaissance_era",
    "Minimalism":                "contemporary",
    "Naive_Art_Primitivism":     "early_modern",
    "New_Realism":               "contemporary",
    "Northern_Renaissance":      "renaissance_era",
    "Pointillism":               "19th_century",
    "Pop_Art":                   "contemporary",
    "Post_Impressionism":        "19th_century",
    "Realism":                   "19th_century",
    "Rococo":                    "18th_century",
    "Romanticism":               "19th_century",
    "Symbolism":                 "19th_century",
    "Synthetic_Cubism":          "early_modern",
    "Ukiyo_e":                   "18th_century",
}

GENRE_TO_STYLE = {
    "Abstract_Expressionism":     ["abstract", "expressionist"],
    "Action_painting":            ["abstract", "expressionist"],
    "Analytical_Cubism":          ["cubist"],
    "Synthetic_Cubism":           ["cubist"],
    "Cubism":                     ["cubist"],
    "Baroque":                    ["baroque"],
    "Early_Renaissance":          ["renaissance"],
    "High_Renaissance":           ["renaissance"],
    "Northern_Renaissance":       ["renaissance"],
    "Mannerism_Late_Renaissance": ["renaissance"],
    "Impressionism":              ["impressionist"],
    "Post_Impressionism":         ["impressionist"],
    "Pointillism":                ["impressionist"],
    "Realism":                    ["realistic"],
    "Contemporary_Realism":       ["realistic"],
    "New_Realism":                ["realistic"],
    "Romanticism":                ["romantic"],
    "Expressionism":              ["expressionist"],
    "Fauvism":                    ["expressionist"],
    "Minimalism":                 ["minimalist"],
    "Pop_Art":                    ["pop_art"],
    "Ukiyo_e":                    ["ukiyo_e"],
    "Rococo":                     ["baroque"],
    "Symbolism":                  ["romantic"],
    "Art_Nouveau_Modern":         ["realistic"],
    "Color_Field_Painting":       ["abstract", "minimalist"],
    "Naive_Art_Primitivism":      ["abstract"],
}
