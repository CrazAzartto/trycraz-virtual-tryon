# Automatic vs manual annotation agreement

Garments compared: 52 (Clothing Dataset images; manual labels by visual inspection).

| Measure | Agreement |
|---|---|
| Garment type, CLIP zero-shot top-1 vs manual (26 classes) | 33/52 = 63% |
| Main colour, k-means name vs manual main colour | 30/52 = 58% |
| Any colour overlap | 33/52 = 63% |

Caveats: small sample; one annotator; several garment types are genuinely ambiguous (polo vs t-shirt, jacket vs blazer); colour names are on a 16-colour palette.

| id | dataset label | manual type | CLIP type | manual colours | auto colours |
|---|---|---|---|---|---|
| cd_001 | Blazer | denim_jacket | denim_jacket | blue | black |
| cd_002 | Blazer | blazer | blazer | navy | black |
| cd_003 | Blazer | leather_jacket | leather_jacket | brown | brown |
| cd_004 | Blazer | blazer | blazer | grey | grey |
| cd_005 | Blazer | blazer | leather_jacket ✗ | maroon | black, maroon |
| cd_006 | Blouse | blouse | blouse | maroon | black, grey |
| cd_007 | Blouse | blouse | shirt_casual ✗ | black | black |
| cd_008 | Blouse | blouse | blouse | white | grey |
| cd_009 | Dress | dress_casual | dress_casual | blue, white | light_blue, grey |
| cd_010 | Dress | dress_casual | dress_formal ✗ | pink | red |
| cd_011 | Dress | dress_formal | polo ✗ | blue | blue, navy |
| cd_012 | Dress | dress_formal | dress_casual ✗ | blue | navy |
| cd_013 | Dress | dress_formal | shirt_formal ✗ | grey | grey |
| cd_014 | Dress | dress_formal | dress_casual ✗ | black | black |
| cd_015 | Hoodie | hoodie | hoodie | blue | blue, navy |
| cd_016 | Hoodie | hoodie | jacket ✗ | white | beige |
| cd_017 | Hoodie | hoodie | hoodie | black | black |
| cd_018 | Longsleeve | sweater | polo ✗ | black | black |
| cd_019 | Longsleeve | sweater | sweatshirt ✗ | grey | grey, beige |
| cd_020 | Longsleeve | sweater | sweater | blue | navy |
| cd_021 | Longsleeve | sweater | sweater | beige | beige |
| cd_022 | Outwear | puffer_jacket | puffer_jacket | blue | navy |
| cd_023 | Outwear | rain_jacket | rain_jacket | maroon | maroon, black |
| cd_024 | Outwear | puffer_jacket | puffer_jacket | black | black |
| cd_025 | Outwear | jacket | puffer_jacket ✗ | grey | grey, black |
| cd_026 | Outwear | jacket | puffer_jacket ✗ | red | maroon |
| cd_027 | Pants | trousers_formal | trousers_formal | grey | grey, black |
| cd_028 | Pants | track_pants | trousers_formal ✗ | grey | grey |
| cd_029 | Pants | trousers_formal | trousers_formal | black | black |
| cd_030 | Pants | jeans | jeans | grey | grey |
| cd_031 | Pants | jeans | chinos ✗ | light_blue | light_blue, grey |
| cd_032 | Polo | polo | polo | light_blue | light_blue |
| cd_033 | Polo | polo | polo | navy | navy, black |
| cd_034 | Polo | polo | polo | green | olive |
| cd_035 | Shirt | shirt_formal | shirt_formal | pink | pink, grey |
| cd_036 | Shirt | shirt_formal | shirt_formal | white | beige, grey |
| cd_037 | Shirt | shirt_casual | shirt_casual | green, white | beige, grey |
| cd_038 | Shirt | shirt_casual | shirt_casual | black, white | grey, black |
| cd_039 | Shirt | shirt_formal | shirt_formal | white | grey |
| cd_040 | Shirt | shirt_casual | shirt_casual | white | white, grey |
| cd_041 | Shorts | shorts | shorts | grey | beige, grey |
| cd_042 | Shorts | shorts | shorts | red | maroon |
| cd_043 | Shorts | shorts | shorts | blue | navy |
| cd_044 | Skirt | skirt | shorts ✗ | blue | grey, navy |
| cd_045 | Skirt | skirt | skirt | black | black |
| cd_046 | T-Shirt | t_shirt | t_shirt | white | white |
| cd_047 | T-Shirt | t_shirt | t_shirt | grey | grey |
| cd_048 | T-Shirt | t_shirt | top ✗ | navy, white | navy, black |
| cd_049 | T-Shirt | t_shirt | top ✗ | grey | grey |
| cd_050 | T-Shirt | t_shirt | top ✗ | yellow | beige, olive |
| cd_051 | Top | top | top | pink | pink |
| cd_052 | Top | top | tank_top ✗ | black | black |