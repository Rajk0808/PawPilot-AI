# Food Analyzer Knowledge Base

## Summary

**Total Items: 60**

### Toxic Foods (30 items)
- **Critical Toxins** (5): Chocolate, Grapes/Raisins, Xylitol, Onions/Garlic/Leeks/Chives, Macadamia Nuts
- **Common Toxins** (5): Avocado, Alcohol, Raw Yeast Dough, Coffee/Caffeine, Excess Salt
- **Additional Toxins** (10): Citrus, Nutmeg, Cherry/Peach/Plum Pits, Cooked Bones, Fatty Foods, Raw Eggs, Raw Meat/Fish, Wild Mushrooms, Moldy Food
- **Cat-Specific Toxins** (5): Lilies, Essential Oils, Milk/Dairy, Tamarind/Cream of Tartar, Raw Fish
- **Critical Miscellaneous** (2): Human Medications, Sago Palm
- **Other** (3): Various common toxins

### Safe Foods (30 items)
- **Primary Safe** (10): Carrots, Blueberries, Chicken, Pumpkin, Green Beans, Apples, Watermelon, Sweet Potatoes, Rice, Yogurt
- **Additional Safe** (10): Bananas, Strawberries, Cucumber, Broccoli, Peanut Butter (xylitol-free), Salmon, Oatmeal, Peas, Eggs, Spinach
- **With USDA Nutrition** (5): Apples, Blueberries, Carrots, Green Beans, Bananas (detailed calorie/portion data)
- **Additional Protein/Veggie/Fruit** (5): Turkey, Beef, Zucchini, Celery, Mango, Cantaloupe, Quinoa, Brown Rice

## Data Quality

✅ **Web-Validated Critical Items:**
- Chocolate: Accurate theobromine levels, onset times, symptoms from Cornell University vet sources
- Grapes/Raisins: Validated tartaric acid research, kidney failure mechanisms from VCA Hospitals
- Xylitol: FDA and ASPCA confirmed toxicity levels
- Onions/Garlic: ASPCA validated Allium toxicity data

## Files Created

```
systems/food_analyzer/knowledge/
├── toxic_foods_critical.json          # 5 items - most dangerous (validated)
├── toxic_foods_common.json            # 5 items - frequently encountered
├── toxic_foods_additional.json        # 10 items - other hazards
├── cat_specific_toxic.json            # 5 items - cat-only toxins (lilies, oils)
├── safe_foods.json                    # 10 items - primary safe foods
├── safe_foods_additional.json         # 10 items - more options
├── safe_foods_nutrition.json          # 5 items - with USDA nutrition data
├── additional_protocols.json          # 10 items - proteins, grains, misc
└── README.md                          # This file
```

## Next Steps

1. ✅ Generate 40 food items (DONE)
2. ⏭️ Create embedding loader script
3. ⏭️ Load into Qdrant vector database
4. ⏭️ Test retrieval accuracy

## JSON Structure

Each item includes:
- Unique ID
- Name and category
- Toxicity level (for toxic) or nutritional value (for safe)
- Affected species
- Symptoms/benefits
- Safe amounts/portions
- Immediate actions/serving suggestions
- **Description field** (what gets embedded for semantic search)
