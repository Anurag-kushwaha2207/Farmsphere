// FarmSphere AI - Agricultural Reference Datasets & Benchmarks
// Based on FAO, ICAR (Indian Council of Agricultural Research) & Global Crop Datasets

export const CROP_DATASET = [
  {
    id: 'wheat',
    name: 'Wheat',
    category: 'Cereal / Rabi',
    scientificName: 'Triticum aestivum',
    optimalN: { min: 80, max: 140, ideal: 120 },
    optimalP: { min: 40, max: 80, ideal: 60 },
    optimalK: { min: 30, max: 60, ideal: 40 },
    optimalPh: { min: 6.0, max: 7.5, ideal: 6.8 },
    optimalTemp: { min: 12, max: 25, ideal: 19 },
    optimalHumidity: { min: 40, max: 70, ideal: 55 },
    optimalRainfall: { min: 350, max: 650, ideal: 500 },
    optimalSoilMoisture: { min: 40, max: 65, ideal: 55 },
    soilTypes: ['Alluvial', 'Loamy', 'Clay Loam'],
    durationDays: 135,
    growthStages: [
      { name: 'Initial / Crown Root Initiation', days: '0-25', kc: 0.4, waterNeed: 'Low' },
      { name: 'Tillering & Vegetative', days: '26-55', kc: 0.8, waterNeed: 'Medium' },
      { name: 'Flowering & Heading', days: '56-85', kc: 1.15, waterNeed: 'High' },
      { name: 'Grain Filling & Milking', days: '86-115', kc: 1.05, waterNeed: 'High' },
      { name: 'Ripening / Maturity', days: '116-135', kc: 0.45, waterNeed: 'Low' }
    ],
    targetYield: { avgTonnesPerHa: 4.5, maxTonnesPerHa: 6.5, avgQuintalPerAcre: 18.2 },
    nutrientReqPerHectare: { N: 120, P: 60, K: 40 },
    splitSchedule: [
      { stage: 'Basal (at Sowing)', N: '50%', P: '100%', K: '100%' },
      { stage: '1st Top Dress (Tillering, 25-30 DAS)', N: '25%', P: '0%', K: '0%' },
      { stage: '2nd Top Dress (Jointing, 45-50 DAS)', N: '25%', P: '0%', K: '0%' }
    ],
    description: 'Premier temperate cereal crop requiring cool vegetative weather and warm sunny ripening conditions.'
  },
  {
    id: 'rice',
    name: 'Rice (Paddy)',
    category: 'Cereal / Kharif',
    scientificName: 'Oryza sativa',
    optimalN: { min: 90, max: 150, ideal: 120 },
    optimalP: { min: 35, max: 70, ideal: 50 },
    optimalK: { min: 40, max: 80, ideal: 60 },
    optimalPh: { min: 5.5, max: 7.0, ideal: 6.2 },
    optimalTemp: { min: 22, max: 35, ideal: 27 },
    optimalHumidity: { min: 70, max: 95, ideal: 82 },
    optimalRainfall: { min: 1000, max: 2500, ideal: 1400 },
    optimalSoilMoisture: { min: 70, max: 95, ideal: 85 },
    soilTypes: ['Clayey', 'Clay Loam', 'Alluvial'],
    durationDays: 140,
    growthStages: [
      { name: 'Transplanting & Nursery', days: '0-20', kc: 1.1, waterNeed: 'Very High' },
      { name: 'Active Tillering', days: '21-50', kc: 1.2, waterNeed: 'Very High' },
      { name: 'Panicle Initiation & Flowering', days: '51-85', kc: 1.35, waterNeed: 'Critical' },
      { name: 'Grain Milking', days: '86-115', kc: 1.15, waterNeed: 'High' },
      { name: 'Maturity & Pre-harvest Drainage', days: '116-140', kc: 0.7, waterNeed: 'Low' }
    ],
    targetYield: { avgTonnesPerHa: 4.8, maxTonnesPerHa: 7.2, avgQuintalPerAcre: 19.5 },
    nutrientReqPerHectare: { N: 130, P: 50, K: 60 },
    splitSchedule: [
      { stage: 'Basal (Transplanting)', N: '33%', P: '100%', K: '50%' },
      { stage: 'Active Tillering (3 weeks)', N: '33%', P: '0%', K: '0%' },
      { stage: 'Panicle Initiation (7-8 weeks)', N: '34%', P: '0%', K: '50%' }
    ],
    description: 'High water-demand staple crop flourishing in sub-tropical moisture and fertile clay-rich soils.'
  },
  {
    id: 'maize',
    name: 'Maize (Corn)',
    category: 'Cereal / Kharif & Rabi',
    scientificName: 'Zea mays',
    optimalN: { min: 70, max: 130, ideal: 100 },
    optimalP: { min: 35, max: 75, ideal: 55 },
    optimalK: { min: 30, max: 65, ideal: 45 },
    optimalPh: { min: 6.0, max: 7.5, ideal: 6.5 },
    optimalTemp: { min: 18, max: 32, ideal: 25 },
    optimalHumidity: { min: 50, max: 80, ideal: 65 },
    optimalRainfall: { min: 500, max: 900, ideal: 650 },
    optimalSoilMoisture: { min: 50, max: 70, ideal: 60 },
    soilTypes: ['Sandy Loam', 'Loamy', 'Black Cotton'],
    durationDays: 110,
    growthStages: [
      { name: 'Seedling & Early Vegetative', days: '0-25', kc: 0.45, waterNeed: 'Low' },
      { name: 'Rapid Vegetative Growth (Knee High)', days: '26-50', kc: 0.85, waterNeed: 'Medium' },
      { name: 'Tasseling & Silking', days: '51-75', kc: 1.2, waterNeed: 'Critical' },
      { name: 'Grain Fill & Dough', days: '76-95', kc: 1.0, waterNeed: 'High' },
      { name: 'Dent & Physiological Maturity', days: '96-110', kc: 0.55, waterNeed: 'Low' }
    ],
    targetYield: { avgTonnesPerHa: 5.2, maxTonnesPerHa: 8.0, avgQuintalPerAcre: 21.0 },
    nutrientReqPerHectare: { N: 120, P: 60, K: 50 },
    splitSchedule: [
      { stage: 'Basal (Sowing)', N: '33%', P: '100%', K: '100%' },
      { stage: 'Knee-High Stage (30 DAS)', N: '33%', P: '0%', K: '0%' },
      { stage: 'Tasseling Stage (50 DAS)', N: '34%', P: '0%', K: '0%' }
    ],
    description: 'Versatile food and fodder crop requiring warm temperatures, sunlight, and well-drained loams.'
  },
  {
    id: 'cotton',
    name: 'Cotton',
    category: 'Commercial / Fibre',
    scientificName: 'Gossypium hirsutum',
    optimalN: { min: 90, max: 140, ideal: 115 },
    optimalP: { min: 40, max: 70, ideal: 55 },
    optimalK: { min: 35, max: 65, ideal: 50 },
    optimalPh: { min: 6.2, max: 8.0, ideal: 7.2 },
    optimalTemp: { min: 21, max: 35, ideal: 28 },
    optimalHumidity: { min: 45, max: 75, ideal: 60 },
    optimalRainfall: { min: 600, max: 1100, ideal: 750 },
    optimalSoilMoisture: { min: 45, max: 65, ideal: 55 },
    soilTypes: ['Black Cotton Soil', 'Clay Loam', 'Deep Alluvial'],
    durationDays: 160,
    growthStages: [
      { name: 'Germination & Emergence', days: '0-20', kc: 0.45, waterNeed: 'Low' },
      { name: 'Square Formation & Squaring', days: '21-60', kc: 0.75, waterNeed: 'Medium' },
      { name: 'Peak Flowering & Boll Setting', days: '61-110', kc: 1.15, waterNeed: 'Critical' },
      { name: 'Boll Development & Opening', days: '111-145', kc: 0.85, waterNeed: 'Medium' },
      { name: 'Maturity & Defoliation', days: '146-160', kc: 0.5, waterNeed: 'Low' }
    ],
    targetYield: { avgTonnesPerHa: 2.3, maxTonnesPerHa: 3.8, avgQuintalPerAcre: 9.3 },
    nutrientReqPerHectare: { N: 100, P: 50, K: 50 },
    splitSchedule: [
      { stage: 'Basal (Sowing)', N: '25%', P: '100%', K: '100%' },
      { stage: 'Square Formation (35 DAS)', N: '50%', P: '0%', K: '0%' },
      { stage: 'Boll Development (65 DAS)', N: '25%', P: '0%', K: '0%' }
    ],
    description: 'Cash crop sensitive to waterlogging, demanding deep moisture-retentive soils and bright sunshine.'
  },
  {
    id: 'chickpea',
    name: 'Chickpea (Gram)',
    category: 'Pulse / Legume',
    scientificName: 'Cicer arietinum',
    optimalN: { min: 20, max: 50, ideal: 30 }, // Legumes fix N
    optimalP: { min: 50, max: 90, ideal: 70 },
    optimalK: { min: 40, max: 80, ideal: 60 },
    optimalPh: { min: 6.0, max: 8.0, ideal: 7.0 },
    optimalTemp: { min: 14, max: 26, ideal: 20 },
    optimalHumidity: { min: 35, max: 60, ideal: 45 },
    optimalRainfall: { min: 300, max: 550, ideal: 400 },
    optimalSoilMoisture: { min: 35, max: 55, ideal: 45 },
    soilTypes: ['Sandy Loam', 'Black Soil', 'Alluvial'],
    durationDays: 110,
    growthStages: [
      { name: 'Emergence & Seedling', days: '0-25', kc: 0.4, waterNeed: 'Low' },
      { name: 'Branching & Vegetative', days: '26-55', kc: 0.7, waterNeed: 'Low to Medium' },
      { name: 'Flowering & Pod Formation', days: '56-85', kc: 1.0, waterNeed: 'Medium' },
      { name: 'Pod Filling & Maturity', days: '86-110', kc: 0.45, waterNeed: 'Low' }
    ],
    targetYield: { avgTonnesPerHa: 1.8, maxTonnesPerHa: 2.8, avgQuintalPerAcre: 7.3 },
    nutrientReqPerHectare: { N: 25, P: 60, K: 30 },
    splitSchedule: [
      { stage: 'Basal (at Sowing)', N: '100%', P: '100%', K: '100%' }
    ],
    description: 'Drought-tolerant leguminous protein crop which fixes atmospheric nitrogen and enriches soil organic matter.'
  },
  {
    id: 'potato',
    name: 'Potato',
    category: 'Tuber / Horticultural',
    scientificName: 'Solanum tuberosum',
    optimalN: { min: 110, max: 170, ideal: 140 },
    optimalP: { min: 60, max: 110, ideal: 85 },
    optimalK: { min: 80, max: 150, ideal: 110 },
    optimalPh: { min: 5.0, max: 6.5, ideal: 5.8 },
    optimalTemp: { min: 15, max: 24, ideal: 19 },
    optimalHumidity: { min: 60, max: 85, ideal: 70 },
    optimalRainfall: { min: 450, max: 750, ideal: 550 },
    optimalSoilMoisture: { min: 60, max: 80, ideal: 70 },
    soilTypes: ['Sandy Loam', 'Loose Alluvial', 'Peaty'],
    durationDays: 100,
    growthStages: [
      { name: 'Sprout & Vegetative', days: '0-25', kc: 0.5, waterNeed: 'Medium' },
      { name: 'Tuber Initiation', days: '26-45', kc: 0.85, waterNeed: 'High' },
      { name: 'Tuber Bulking', days: '46-80', kc: 1.15, waterNeed: 'Critical' },
      { name: 'Tuber Maturation & Skin Setting', days: '81-100', kc: 0.7, waterNeed: 'Low' }
    ],
    targetYield: { avgTonnesPerHa: 24.0, maxTonnesPerHa: 38.0, avgQuintalPerAcre: 97.0 },
    nutrientReqPerHectare: { N: 150, P: 80, K: 120 },
    splitSchedule: [
      { stage: 'Basal (at Planting)', N: '50%', P: '100%', K: '50%' },
      { stage: 'At Earthing Up (30-35 DAP)', N: '50%', P: '0%', K: '50%' }
    ],
    description: 'High-value tuber requiring loose friable acidic-to-neutral soils with steady moisture during bulking.'
  }
];

// Reference Soil Fertility Classifications (ICAR / FAO benchmarks)
export const SOIL_STANDARDS = {
  nitrogen: {
    low: { max: 50, label: 'Low (< 50 mg/kg)', status: 'Deficient' },
    medium: { min: 50, max: 100, label: 'Medium (50 - 100 mg/kg)', status: 'Optimal' },
    high: { min: 100, label: 'High (> 100 mg/kg)', status: 'Surplus' }
  },
  phosphorus: {
    low: { max: 30, label: 'Low (< 30 mg/kg)', status: 'Deficient' },
    medium: { min: 30, max: 60, label: 'Medium (30 - 60 mg/kg)', status: 'Optimal' },
    high: { min: 60, label: 'High (> 60 mg/kg)', status: 'Surplus' }
  },
  potassium: {
    low: { max: 35, label: 'Low (< 35 mg/kg)', status: 'Deficient' },
    medium: { min: 35, max: 80, label: 'Medium (35 - 80 mg/kg)', status: 'Optimal' },
    high: { min: 80, label: 'High (> 80 mg/kg)', status: 'Surplus' }
  },
  ph: {
    stronglyAcidic: { max: 5.5, label: 'Strongly Acidic (< 5.5)', remedy: 'Apply Agricultural Lime (CaCO3) @ 1.5 - 2.5 t/ha' },
    moderatelyAcidic: { min: 5.5, max: 6.2, label: 'Slightly Acidic (5.5 - 6.2)', remedy: 'Add farmyard manure & mild dolomitic lime' },
    neutral: { min: 6.2, max: 7.5, label: 'Neutral & Optimal (6.2 - 7.5)', remedy: 'Ideal nutrient availability; maintain organic matter' },
    alkaline: { min: 7.5, max: 8.5, label: 'Moderately Alkaline (7.5 - 8.5)', remedy: 'Apply Gypsum (CaSO4·2H2O) @ 1.0 - 2.0 t/ha & green manuring' },
    stronglyAlkaline: { min: 8.5, label: 'Strongly Alkaline / Sodic (> 8.5)', remedy: 'Deep leaching with gypsum and elemental sulfur application' }
  },
  organicCarbon: {
    low: { max: 0.5, label: 'Low (< 0.5%)', status: 'Poor microbial activity; apply 10 t/ha FYM or vermicompost' },
    medium: { min: 0.5, max: 0.75, label: 'Medium (0.5% - 0.75%)', status: 'Moderate fertility; retain crop residues' },
    high: { min: 0.75, label: 'High (> 0.75%)', status: 'Excellent organic soil biology' }
  },
  electricalConductivity: { // dS/m
    normal: { max: 1.0, label: 'Normal (< 1.0 dS/m)', status: 'Non-saline, ideal for all crops' },
    slightSalinity: { min: 1.0, max: 2.0, label: 'Slight Salinity (1.0 - 2.0 dS/m)', status: 'Minor yield risk for sensitive crops' },
    saline: { min: 2.0, label: 'Saline (> 2.0 dS/m)', status: 'Critical salinity; install tile drainage & leach with good quality water' }
  }
};

// Commercial Fertilizer Nutritive Contents
export const COMMERCIAL_FERTILIZERS = [
  { id: 'urea', name: 'Urea', N: 46, P: 0, K: 0, form: 'Granular', desc: 'Primary fast-release nitrogen source' },
  { id: 'dap', name: 'DAP (Di-Ammonium Phosphate)', N: 18, P: 46, K: 0, form: 'Granular', desc: 'High-phosphorus basal fertilizer' },
  { id: 'mop', name: 'MOP (Muriate of Potash / KCl)', N: 0, P: 0, K: 60, form: 'Powder/Granule', desc: 'Standard potassium chloride' },
  { id: 'ssp', name: 'SSP (Single Super Phosphate)', N: 0, P: 16, K: 0, S: 11, Ca: 19, form: 'Granular', desc: 'Provides Phosphorus + Sulphur + Calcium' },
  { id: 'npk_complex', name: 'NPK 12:32:16', N: 12, P: 32, K: 16, form: 'Complex Granular', desc: 'Balanced starter fertilizer' },
  { id: 'zinc_sulphate', name: 'Zinc Sulphate (21% Zn)', Zn: 21, S: 10, form: 'Micronutrient', desc: 'Corrects Khaira / zinc deficiency' }
];

// Leaf Disease Reference Library
export const CROP_DISEASES = [
  {
    id: 'wheat_rust',
    crop: 'Wheat',
    diseaseName: 'Yellow / Stripe Rust',
    pathogen: 'Puccinia striiformis f. sp. tritici',
    severity: 'High',
    symptoms: 'Yellow to orange-yellow pustules arranged in linear stripes along leaf veins; stunted spikes.',
    organicRemedy: 'Neem seed kernel extract (NSKE 5%) spray; cultivate resistant cultivars like HD-2967, PBW-550.',
    chemicalControl: 'Propiconazole 25% EC (Tilt) @ 1 ml/litre water or Tebuconazole 25.9% EC @ 1.25 ml/litre.',
    favorableConditions: 'High humidity (>85%), temperatures between 10°C - 20°C, morning dew.'
  },
  {
    id: 'rice_blast',
    crop: 'Rice',
    diseaseName: 'Rice Leaf & Neck Blast',
    pathogen: 'Magnaporthe oryzae (Pyricularia oryzae)',
    severity: 'Critical',
    symptoms: 'Spindle-shaped or diamond-shaped lesions with grey/white centers and reddish-brown borders; node rotting.',
    organicRemedy: 'Seed treatment with Pseudomonas fluorescens @ 10g/kg seed; balanced nitrogen (avoid excessive urea).',
    chemicalControl: 'Tricyclazole 75% WP @ 0.6 g/litre or Kasugamycin 3% SL @ 2.5 ml/litre water.',
    favorableConditions: 'Cloudy overcast days, high night humidity, temperatures around 24°C - 28°C.'
  },
  {
    id: 'potato_late_blight',
    crop: 'Potato',
    diseaseName: 'Late Blight',
    pathogen: 'Phytophthora infestans',
    severity: 'Severe',
    symptoms: 'Water-soaked irregular pale green/brown lesions on leaf margins; white fungal fuzz on lower leaf surface.',
    organicRemedy: 'Trichoderma viride foliar spray; avoid overhead sprinkler irrigation; destroy infected volunteer plants.',
    chemicalControl: 'Mancozeb 75% WP (Dithane M-45) @ 2.5 g/L as prophylactic, followed by Cymoxanil + Mancozeb @ 2.5 g/L.',
    favorableConditions: 'Prolonged cold foggy weather, temp 12°C - 22°C with RH > 90%.'
  },
  {
    id: 'maize_leaf_blight',
    crop: 'Maize',
    diseaseName: 'Maydis Leaf Blight',
    pathogen: 'Bipolaris maydis',
    severity: 'Moderate',
    symptoms: 'Elongated rectangular light brown lesions confined between leaf veins.',
    organicRemedy: 'Crop rotation with non-host legumes; deep plowing of infected crop debris.',
    chemicalControl: 'Spray Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/litre water.',
    favorableConditions: 'Warm humid weather (24°C - 30°C) with frequent rains.'
  }
];
