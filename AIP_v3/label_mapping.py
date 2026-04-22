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

TECHNIQUE_TAGS = [
    "thick_brushwork", "fine_detail", "loose_brushwork",
    "pointillist_dots", "flat_color", "sketch_like", "layered_depth"
]

SUBJECT_TAGS = [
    "portrait", "landscape", "still_life", "religious_mythological",
    "historical_battle", "everyday_life", "sea_water", "urban_city",
    "animals", "nude", "abstract_nonrepresentational"
]

ALL_TAGS = STYLE_TAGS + MOOD_TAGS + ELEMENT_TAGS + TECHNIQUE_TAGS + SUBJECT_TAGS

# Maps each WikiArt genre folder name to a list of tags from ALL_TAGS
GENRE_TO_TAGS = {
    "Abstract_Expressionism": [
        "abstract",
        "energetic", "dramatic", "mysterious",
        "vivid_colors", "high_contrast", "rich_texture",
        "thick_brushwork", "loose_brushwork",
        "abstract_nonrepresentational"
    ],
    "Action_painting": [
        "abstract",
        "energetic", "dramatic",
        "vivid_colors", "high_contrast", "rich_texture",
        "thick_brushwork", "loose_brushwork",
        "abstract_nonrepresentational"
    ],
    "Analytical_Cubism": [
        "cubist",
        "mysterious",
        "muted_tones", "geometric_shapes",
        "flat_color", "sketch_like",
        "portrait", "still_life"
    ],
    "Art_Nouveau_Modern": [
        "abstract",
        "mysterious", "serene",
        "vivid_colors", "warm_tones", "rich_texture",
        "fine_detail", "layered_depth",
        "portrait", "everyday_life"
    ],
    "Baroque": [
        "baroque",
        "dramatic", "spiritual", "dark",
        "high_contrast", "dark_shadows", "warm_tones", "rich_texture",
        "fine_detail", "layered_depth",
        "religious_mythological", "portrait", "historical_battle"
    ],
    "Color_Field_Painting": [
        "abstract", "minimalist",
        "serene", "peaceful",
        "vivid_colors", "geometric_shapes",
        "flat_color",
        "abstract_nonrepresentational"
    ],
    "Contemporary_Realism": [
        "realistic",
        "peaceful", "melancholic",
        "muted_tones", "soft_light",
        "fine_detail", "layered_depth",
        "portrait", "everyday_life", "landscape"
    ],
    "Cubism": [
        "cubist",
        "mysterious", "energetic",
        "geometric_shapes", "vivid_colors",
        "flat_color", "sketch_like",
        "portrait", "still_life"
    ],
    "Early_Renaissance": [
        "renaissance",
        "spiritual", "peaceful", "serene",
        "soft_light", "warm_tones",
        "fine_detail", "layered_depth",
        "religious_mythological", "portrait"
    ],
    "Expressionism": [
        "expressionist",
        "dramatic", "dark", "tense", "melancholic",
        "vivid_colors", "high_contrast",
        "thick_brushwork", "loose_brushwork",
        "portrait", "landscape", "everyday_life"
    ],
    "Fauvism": [
        "expressionist",
        "joyful", "energetic",
        "vivid_colors", "warm_tones", "high_contrast",
        "thick_brushwork", "loose_brushwork", "flat_color",
        "landscape", "portrait", "everyday_life"
    ],
    "High_Renaissance": [
        "renaissance",
        "spiritual", "dramatic", "serene",
        "soft_light", "warm_tones", "rich_texture",
        "fine_detail", "layered_depth",
        "religious_mythological", "portrait", "historical_battle"
    ],
    "Impressionism": [
        "impressionist",
        "peaceful", "joyful", "serene",
        "soft_light", "warm_tones", "vivid_colors",
        "loose_brushwork", "thick_brushwork",
        "landscape", "everyday_life", "portrait", "sea_water"
    ],
    "Mannerism_Late_Renaissance": [
        "renaissance",
        "mysterious", "dramatic", "tense",
        "warm_tones", "high_contrast", "rich_texture",
        "fine_detail", "layered_depth",
        "religious_mythological", "portrait", "historical_battle"
    ],
    "Minimalism": [
        "minimalist", "abstract",
        "serene", "peaceful",
        "muted_tones", "geometric_shapes",
        "flat_color",
        "abstract_nonrepresentational"
    ],
    "Naive_Art_Primitivism": [
        "abstract",
        "joyful", "peaceful",
        "vivid_colors",
        "flat_color", "sketch_like",
        "landscape", "everyday_life", "animals"
    ],
    "New_Realism": [
        "realistic",
        "energetic", "joyful",
        "vivid_colors", "high_contrast",
        "fine_detail", "flat_color",
        "everyday_life", "urban_city"
    ],
    "Northern_Renaissance": [
        "renaissance",
        "spiritual", "dark", "mysterious",
        "high_contrast", "dark_shadows", "rich_texture", "warm_tones",
        "fine_detail", "layered_depth",
        "religious_mythological", "portrait", "everyday_life"
    ],
    "Pointillism": [
        "impressionist",
        "peaceful", "joyful", "serene",
        "vivid_colors", "soft_light", "warm_tones",
        "pointillist_dots",
        "landscape", "everyday_life", "portrait"
    ],
    "Pop_Art": [
        "pop_art",
        "joyful", "energetic",
        "vivid_colors", "high_contrast", "geometric_shapes",
        "flat_color",
        "everyday_life", "urban_city", "portrait"
    ],
    "Post_Impressionism": [
        "impressionist", "expressionist",
        "melancholic", "mysterious", "dramatic",
        "vivid_colors", "warm_tones", "rich_texture", "high_contrast",
        "thick_brushwork", "loose_brushwork",
        "landscape", "portrait", "everyday_life"
    ],
    "Realism": [
        "realistic",
        "melancholic", "peaceful", "dark",
        "muted_tones", "soft_light", "rich_texture",
        "fine_detail", "layered_depth",
        "portrait", "everyday_life", "landscape", "historical_battle"
    ],
    "Rococo": [
        "baroque",
        "joyful", "serene", "peaceful",
        "soft_light", "warm_tones", "vivid_colors", "rich_texture",
        "fine_detail", "layered_depth",
        "portrait", "everyday_life", "nude", "landscape"
    ],
    "Romanticism": [
        "romantic",
        "dramatic", "melancholic", "mysterious",
        "high_contrast", "dark_shadows", "warm_tones", "rich_texture",
        "fine_detail", "loose_brushwork", "layered_depth",
        "landscape", "historical_battle", "portrait", "sea_water"
    ],
    "Symbolism": [
        "romantic", "expressionist",
        "mysterious", "melancholic", "dark", "spiritual",
        "muted_tones", "dark_shadows", "rich_texture", "cool_tones",
        "fine_detail", "layered_depth",
        "portrait", "religious_mythological", "landscape"
    ],
    "Synthetic_Cubism": [
        "cubist", "abstract",
        "energetic", "mysterious",
        "vivid_colors", "geometric_shapes", "high_contrast",
        "flat_color", "sketch_like",
        "portrait", "still_life", "abstract_nonrepresentational"
    ],
    "Ukiyo_e": [
        "ukiyo_e",
        "peaceful", "serene", "melancholic",
        "vivid_colors", "muted_tones",
        "fine_detail", "flat_color",
        "portrait", "landscape", "everyday_life", "sea_water"
    ],
}
