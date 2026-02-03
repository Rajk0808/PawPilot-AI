from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
     
class KeyWordExtractor:
    def __init__(self):
        self.default_strategy = "default"
        # ===== DISEASES & HEALTH CONDITIONS (EXPANDED) =====
        self.diseases_keywords = set('''
            diseases illness ringworm mange cancer tumor cyst infection hot spot dermatitis yeast 
            fleas ticks mites rash redness hair loss alopecia bald spots swelling bumps lumps scabs 
            bleeding pus oozing crusty flaky skin dandruff inflammation lesion wound injury cut scratch 
            bite mark itchy scratching licking painful hurting vet emergency contagious dangerous treatment 
            diagnosis fever chills lethargy limp lameness arthritis joint pain stiffness paralysis seizure 
            convulsion stroke collapse unconscious unresponsive breathing difficulty cough wheeze sneeze 
            nasal discharge eye discharge conjunctivitis blindness ear infection mite deafness ear canal 
            ulcer abscess polyp growth mass nodule hernia prolapse incontinence diarrhea constipation 
            bloat gastric dilatation pancreatitis liver disease kidney disease diabetes hypoglycemia 
            anemia leukemia lymphoma heartworm hookworm tapeworm roundworm giardia coccidia fungal 
            bacterial viral parasitic allergies food allergies environmental allergies contact dermatitis 
            atopy eczema psoriasis seborrhea keratosis wart papilloma melanoma mast cell tumor 
            hemangiosarcoma osteosarcoma lymphosarcoma fibrosarcoma hemangioma lipoma fibroma myoma 
            angioma granuloma abscess fistula cyst polyp nodule growth lump mass swelling edema 
            effusion fluid accumulation ascites pleural effusion pericardial effusion subcutaneous 
            hematoma bruising contusion fracture break dislocation sprain strain tear rupture avulsion 
            degenerative joint disease osteoarthritis bone disease rickets hypocalcemia hypercalcemia 
            metabolic bone disease spinal cord injury nerve damage paralysis ataxia tremor incoordination 
            behavioral changes aggression anxiety depression apathy lethargy hyperactivity ADHD OCD 
            compulsive disorder phobia fear anxiety separation anxiety stress burnout exhaustion 
            malnutrition obesity underweight overweight emaciation cachexia muscle wasting atrophy 
            hypertrophy muscle weakness muscle pain cramps spasms fasciculation trembling shaking 
            pain chronic pain neuropathic pain inflammatory pain referred pain somatic pain visceral 
            pain periostitis myositis fascitis tendonitis bursitis synovitis arthralgia myalgia 
            gastroenteritis enteric disease enteritis colitis hepatitis nephritis cystitis urethra 
            urinary incontinence fecal incontinence dysuria hematuria glycosuria proteinuria ketonuria 
            anaerobic myonecrosis necrosis tissue death sloughing eschar gangrene sepsis septicemia 
            shock toxemia toxins bacterial toxins enterotoxin Clostridium toxin staphylococcal toxin 
            anaphylaxis anaphylactic reaction allergic reaction atopic dermatitis atopic dog atopic cat 
            feline rhinotracheitis calicivirus panleukopenia FIV feline leukemia feline infectious disease 
            canine distemper canine parvovirus canine coronavirus kennel cough bordetella parainfluenza 
            leptospirosis brucellosis toxoplasmosis blastomycosis histoplasmosis coccidioidomycosis 
            cryptococcosis aspergillosis candidiasis malassezia otitis externa otitis media otitis interna 
            external ear canal infection middle ear infection inner ear infection vestibular disease 
            horner syndrome third eyelid prolapse cherry eye nictitans haw prolapse entropion ectropion 
            distichiasis trichiasis ptosis micropsia macropsia anisocoria mydriasis miosis 
            dental disease periodontal disease gingivitis stomatitis gingivitis calculus tartar 
            tooth resorption orthodontic malocclusion bite misalignment anterior cross bite posterior 
            tooth fracture tooth decay dental caries root exposure root canal inflammation pulpitis 
            oral ulcer oral erosion necrotizing ulcerative gingivitis NUGH feline stomatitis feline 
            salivary gland disorder salivary calculus salivary cyst ptyalocele salivary fistula 
            mucositis esophagitis esophageal stricture esophageal megaesophagus achalasia 
            vomiting hematemesis projectile vomiting bilious vomiting regurgitation dysphagia 
            difficulty swallowing food sticking aspiration pneumonia aspiration risk swallow test 
            gastroesophageal reflux GERD reflux disease reflux esophagitis Barrett esophagus 
            pyloric stenosis pylorospasm gastric outflow obstruction intestinal obstruction ileus 
            megacolon toxic megacolon acute abdomen abdominal pain peritonitis peritoneal effusion 
            ascites abdominal distension abdominal bloating abdominal fluid accumulation 
            hepatic lipidosis hepatic encephalopathy hepatic failure liver encephalopathy 
            portosystemic shunt PSS portal hypertension cirrhosis fibrosis steatosis 
            pancreatitis acute pancreatitis chronic pancreatitis pancreatic insufficiency 
            exocrine pancreatic insufficiency EPI endocrine pancreatic disease diabetes mellitus 
            diabetic ketoacidosis DKA hyperglycemia hypoglycemia insulin resistance 
            kidney disease chronic kidney disease CKD acute kidney injury AKI renal failure 
            uremia uremic toxins azotemia creatinine elevation BUN blood urea nitrogen 
            proteinuria proteinuric kidney disease glomerulonephritis pyelonephritis cystitis 
            uroliths urinary stones kidney stones struvite stones oxalate stones uric acid stones 
            urinary blockage urinary obstruction urinary retention urinary incontinence dysuria 
            hematuria pyuria urinary tract infection lower urinary tract disease LUTD FUS 
            bladder infection bladder inflammation cystitis bladder stones bladder tumor 
            prostate prostate enlargement benign prostatic hyperplasia BPH prostatitis 
            prostatic cancer prostatic fluid prostatic infection prostatic abscess 
            testicular testicular atrophy testicular tumor testicular torsion testicular trauma 
            cryptorchidism retained testicle undescended testicle orchitis epididymitis 
            penile phimosis paraphimosis penile fracture penile trauma penile infection 
            preputial infection preputial discharge penile discharge abnormal discharge 
            ovarian ovarian tumor ovarian cyst ovarian remnant syndrome post ovariohysterectomy 
            uterine uterine infection metritis pyometra uterine rupture uterine torsion 
            uterine tumor endometrial cancer endometrial polyp endometrial cyst endometrial hyperplasia 
            mammary mammary gland infection mastitis mammary cancer mammary tumor mammary gland trauma 
            vulvar infection vulvitis vulvar discharge vulvar tumor vulvar trauma vulvar dermatitis 
            vaginal infection vaginitis vaginal discharge vaginal tumor vaginal trauma vaginal stricture 
            vaginal anomaly hymenal stricture vaginal agenesis agenesis uteri 
            hormonal imbalance endocrine disorder thyroid hyperthyroidism hypothyroidism thyroiditis 
            thyroid cancer thyroid nodule thyroid cyst thyroid inflammation thyroid dysfunction 
            parathyroid parathyroid hyperplasia parathyroid adenoma parathyroid carcinoma 
            adrenal adrenal insufficiency Addison disease adrenal excess Cushing disease 
            pituitary pituitary tumor pituitary dwarfism growth hormone deficiency 
            pituitary hyperfunction pituitary hypofunction diabetes insipidus 
            bone disease osteodystrophy nutritional secondary hyperparathyroidism 
            rickets hypocalcemia hypercalcemia calcium phosphorus mineral imbalance 
            carpal flexural deformity angular limb deformity leg deformity panosteitis 
            hip dysplasia elbow dysplasia OCD osteochondritis dissecans osteochondritis 
            osteochondrosis hypertrophic osteodystrophy HOD retained cartilage core phenomenon 
            panosteitis osteonecrosis legg perthes disease slipped capital femoral epiphysis 
            luxating patella patellar luxation genu valgum genu varum limb valgus limb varus 
            ligament tear ACL tear cruciate ligament tear anterior cruciate ligament posterior 
            meniscal tear meniscus tear collateral ligament tear medial collateral lateral collateral 
            hip dysplasia hip joint disease hip subluxation hip luxation hip osteoarthritis 
            shoulder luxation shoulder pain shoulder lameness rotator cuff tear shoulder instability 
            elbow dysplasia elbow pain elbow lameness elbow joint disease elbow osteoarthritis 
            carpal carpal pain carpal sprain carpal instability carpal hyperextension 
            tarsal tarsal pain tarsal sprain tarsal instability tarsal hyperextension 
            muscle muscle tear muscle strain muscle sprain muscle rupture muscle inflammation 
            myositis myositis ossificans heterotopic ossification muscle calcification 
            nerve nerve injury nerve damage peripheral neuropathy nerve compression nerve entrapment 
            intervertebral disc disease IVDD disc herniation disc extrusion disc protrusion 
            fibrocartilage embolism FCE spinal cord compression myelopathy 
            cervical myelopathy thoracic myelopathy lumbar myelopathy cauda equina syndrome 
            lumbosacral stenosis lumbosacral disease foraminal stenosis canal stenosis 
            atlanto axial instability wobbly kitten feline hyperesthesia syndrome FHS 
            pain tail syndrome degenerative myelopathy DM canine degenerative myelopathy 
            fibrocartilagenous embolism spinal cord ischemia spinal cord infarction 
            spinal stroke spinal cord trauma spinal fracture spinal luxation spinal subluxation 
            spinal fusion spinal arthrodesis vertebral body fracture vertebral body collapse 
            spondylosis spondylarthrosis spondylitic changes osseous metaplasia 
            exostosis osteophyte bone spur new bone formation hypertrophic changes 
            neoplasia malignancy cancer oncology tumor growth metastasis metastatic disease 
            primary tumor secondary tumor lymph node involvement systemic disease disseminated 
            bone marrow disease bone marrow suppression bone marrow necrosis bone marrow fibrosis 
            anemia aplastic anemia hemolytic anemia iron deficiency anemia macrocytic microcytic 
            thrombocytopenia thrombocytosis bleeding coagulopathy clotting disorder 
            von willebrand disease hemophilia Factor deficiency platelet dysfunction 
            disseminated intravascular coagulation DIC thromboembolism thrombosis blood clot 
            leukopenia leukocytosis leukemia lymphoma myeloma plasma cell tumor 
            immunosuppression immunodeficiency FIV feline leukemia FeLV retroviruses 
            vaccination vaccination response immunization immune response antibody titer 
            hypersensitivity reaction delayed hypersensitivity contact dermatitis 
            chemical burn corrosive injury caustic injury irritant dermatitis irritant reaction 
        '''.split())
        
        # ===== FULL BODY HEALTH SCAN =====
        self.health_scan_keywords = set('''
            health scan body check physical exam appearance condition posture gait walk stride 
            limp lameness limping hopping three legged bearing weight weight bearing joint stiffness 
            flexibility range motion mobility movement locomotion exercise tolerance fatigue weakness 
            lethargy sluggish slow inactive lethargic energy level vitality vigor stamina endurance 
            obese obesity overweight underweight emaciated thin skinny bony ribs spine visible prominent 
            muscle definition muscle loss muscle wasting atrophy muscular build physique frame size 
            coat condition fur coat hair shine luster quality matting tangles mat hair loss shedding 
            excessive shedding alopecia bald patch fur quality dull dry oily greasy flaky dandruff 
            skin condition skin health pigmentation pigment changes discoloration pale pale gums 
            inflamed gums gum disease dental plaque tartar teeth condition tooth health teeth loss 
            jaw alignment bite occlusion malocclusion alignment underbite overbite crossbite open bite 
            breathing pattern respiratory rate respiration shortness breath dyspnea tachypnea apnea 
            coughing sneezing sniffling nasal discharge nasal congestion runny nose postnasal drip 
            heart rate pulse rhythm regular irregular arrhythmia murmur heart sounds cardiac abnormality 
            abdominal bloating distension ascites fluid accumulation hardness softness tenderness 
            lumps bumps masses tumors swelling edema inflammation joint swelling effusion fluid 
            eye appearance eyes bright alert dull cloudy cataracts lens opacity pupil size reactivity 
            discharge crusting redness conjunctival injection third eyelid prolapse haw cherry eye 
            ear appearance ear position ears alert droopy floppy standing erect ear discharge earwax 
            ear canal inflammation redness infection odor foul smell parasites mites fleas ticks 
            skin parasites external parasites lice mange mites demodex sarcoptes otodectic 
            paw pads color texture cracks calluses hyperkeratosis paw pad injuries wounds bleeding 
            nails nail color condition overgrown splitting breaking curved deformed brittle strong 
            tail appearance tail position tail carriage tail wagging limp drooping tucked raised 
            anal glands anal sacs expression scooting dragging bottom butt rubbing perianal area 
            genital area appearance discharge vulva penile preputial abnormalities 
            lymph nodes palpable enlarged swollen submandibular axillary inguinal abdominal 
            temperature fever hot cool normal body condition score BCS fat deposits rib cage feeling 
            spine palpation abdominal palpation pain response guarding tenseness relaxation 
            behavior demeanor attitude disposition temperament personality mood happy sad depressed 
            anxious fearful confident bold timid shy social unsocial reactive responsive interactive 
            senior posture aged elderly geriatric young puppy juvenile adult prime middle aged 
            overall assessment assessment score overall health overall wellness general condition 
            visible abnormalities obvious abnormalities physical abnormalities deformities 
            asymmetry asymmetrical appearance uneven uneven gait uneven size uneven development 
            hydration status dehydration skin turgor mucous membrane color capillary refill time 
            baseline comparison comparison baseline previous history historical trend 
            alert responsive conscious awareness mentation alertness orientation responsiveness 
            pain localization pain location pain palpation tenderness response reaction 
            reflex reflex test reflex response normal reflex abnormal reflex exaggerated reflex 
            coordination incoordination ataxia balance proprioception motor control movement control 
            sensory sensation sensory response sensory deficit neuropathy nerve dysfunction 
            tremor trembling shaking muscle fasciculation spasm spasmodic involuntary movement 
            discharge discharge present discharge type discharge color discharge consistency 
            odor smell bad smell foul smell healthy normal smell 
        '''.split())
        
        # ===== WEIGHT TRACKER & BODY CONDITION =====
        self.weight_keywords = set('''
            weight obesity overweight underweight ideal weight gain loss track monitor growth growth rate 
            body condition score BCS thin skinny bony ribs visible spine visible prominent rib palpable 
            abdominal tuck waist definition waist circumference girth measurement scale weigh in 
            growth chart development stage life stage age maturity adult senior elderly puppy juvenile 
            adolescent prime adult mature growth puppy growth curve growth rate normal abnormal rapid slow 
            stunted dwarfism gigantism obesity obese overweight chubby chunky pudgy rotund round 
            barrel chested potbellied paunch spare tire abdominal fat fat deposits body fat percentage 
            muscle tone muscle mass muscle wasting lean athletic physique build frame size large small 
            breed standard breed specific ideal weight range target weight goal healthy BMI body mass index 
            caloric intake calories burned exercise activity level sedentary inactive moderate active 
            hyperactive overactive underactive lethargy fatigue weight management diet plan nutrition 
            food intake eating habits meal portions portion control feeding schedule meal frequency 
            metabolism metabolic rate metabolic disorder thyroid hyperthyroidism hypothyroidism 
            metabolism slow fast normal metabolic dysfunction metabolic disease endocrine hormonal 
            menopause spay neuter altered intact reproductive status neutered spayed castrated 
            supplement feeding treat intake snack table food human food people food scraps 
            density muscle fat ratio body composition muscle fat ratio lean body mass fat body mass 
            before after weight progress progress tracking trend trajectory improvement decline 
            target goal milestone achievement success failure weight plateau plateau breaker 
            rib cage feeling ribs palpable ribs prominent rib visibility rib count rib cage movement 
            spine palpability spine visibility spinal process prominence vertebrae feeling 
            hip prominence hip visibility hip bone feeling hip structure hip anatomy 
            abdominal appearance abdominal tuck tucked abdomen hanging abdomen pendulous abdomen 
            waist appearance waist definition waist narrowing waist widening hourglass figure 
            roundness roundness appearance abdomen shape abdominal profile side profile 
            thigh appearance thigh muscle thigh fat thigh tone leg appearance leg muscle 
            shoulder appearance shoulder blade scapula prominence scapula visibility 
            overall body shape body shape assessment shape change shape transformation 
            maturity assessment physical maturity skeletal maturity growth stage growth period 
            growth completion growth finished growth ongoing growth rate growth velocity 
            feeding frequency meal frequency feeding times feeding schedule feeding amount 
            caloric needs daily caloric needs daily requirements age appropriate portion 
            life stage food senior food adult food puppy food juvenile food 
            weight loss plan weight loss program weight loss success weight loss goal 
            weight gain weight gain plan weight gain goal weight gain success 
            plateau weight plateau appetite changes appetite suppression appetite increase 
            activity increase activity decrease activity level change mobility improvement 
            thyroid test thyroid function thyroid level TSH test thyroid hormone 
            metabolic panel metabolic testing blood work blood test chemistry panel 
            abnormal weight change rapid weight change sudden weight loss sudden weight gain 
            trend analysis trending trending up trending down consistent change 
        '''.split())
        
        # ===== PET FOOD IMAGE ANALYSIS =====
        self.food_keywords = set('''
            food meal eat eating diet nutrition nutrient ingredient ingredients recipe cooked raw fresh 
            kibble dry food wet food canned food frozen food fresh food home cooked commercial brand 
            meat beef chicken pork fish lamb turkey duck rabbit venison game meat organ meat offal liver 
            kidney heart lung brain intestine tongue tripe muscle meat protein carbohydrate fat 
            fiber ash moisture content calorie kcal caloric content nutrition facts label ingredients 
            grain corn wheat soy soybean rice oat barley rye bran germ meal flour starch carbs grains 
            vegetables vegetable spinach broccoli carrot sweet potato potato pumpkin squash pea bean 
            legume lentil chickpea tofu soy fruit apple banana orange blueberry strawberry watermelon 
            grapes raisins cranberry avocado coconut pineapple mango peach apricot prune fig date 
            toxic poison poisonous onion garlic chive leek allium xylitol chocolate cocoa theobromine 
            caffeine coffee tea raisin grape avocado macadamia nut walnut mushroom salt sodium sugar 
            artificial sweetener artificial flavoring artificial coloring preservative BHA BHT ethoxyquin 
            by product by products meal meat by product corn by product soy by product wheat by product 
            filler cheap quality low quality premium quality high quality grain free limited ingredient 
            novel protein hypoallergenic allergenic allergen allergens food allergy food sensitivity 
            digestibility digestible indigestible easily digested hard to digest upset stomach GI upset 
            portion size serving size recommendation daily caloric intake recommended intake calories 
            nutritional balance nutritionally balanced complete nutrition aafco certified aafco approved 
            life stage puppy adult senior prescription diet therapeutic diet medical diet prescription 
            senior diet small breed large breed giant breed toy breed miniature breed specific formula 
            weight management sensitive skin digestive health joint health kidney health liver health 
            cardiac health heart health diabetes diet urinary health UTI health allergy diet limited 
            ingredient novel protein organic natural holistic whole food raw diet barf biologically 
            appropriate raw food vegan vegetarian supplement vitamin mineral probiotic prebiotic 
            omega fatty acid omega 3 omega 6 antioxidant antioxidants glucosamine chondroitin 
            joint support mobility turmeric green tea cranberry blueberry fish oil salmon oil coconut 
            meal alternatives substitute substitute ingredient replacement swap healthier option 
            safe food unsafe food people food human food toxic food dangerous food risky food 
            freshness expiration date shelf life storage preparation cooking temperature doneness 
            raw meat bacteria salmonella E coli listeria parasites parasitic cysts freezing thawing 
            cross contamination food safety hygiene sanitation contaminated spoiled rotten bad fresh 
            quantity amount bowl feeding freefeed free feeding scheduled feeding meal time snack 
            treat treat time treat reward training treat high value treat low calorie treat 
            appetite loss decreased appetite anorexia increased appetite polyphagia normal appetite 
            picky eater finicky selective eater food refusal food rejection food preferences taste 
            texture kibble size small kibble large kibble medium kibble palatability palatably 
            digestibility stool quality firm stool loose stool diarrhea constipation soft stool 
            smell odor odorous foul smelling fresh feces vomit vomiting regurgitation 
            hydration dehydration water intake drinking habits water bowl fresh water 
            nutrients essential amino acids amino acid taurine carnitine complete protein 
            protein quality biological value digestibility protein digestibility amino acid profile 
            fat content essential fatty acids arachidonic acid DHA EPA linoleic acid linolenic acid 
            vitamin vitamin A vitamin D vitamin E vitamin K vitamin B vitamin B12 folate thiamine 
            mineral phosphorus calcium calcium phosphorus ratio magnesium zinc iron copper iodine 
            deficiency excess toxicity supplementation fortified 
        '''.split())
        
        # ===== PACKAGED PRODUCT SCANNER =====
        self.packaged_product_keywords = set('''
            product scan scanner barcode label packaging ingredient ingredients dry food kibble 
            wet food canned food treat treats snack snacks supplement supplements vitamin minerals 
            shampoo conditioner grooming soap spray deodorant cologne perfume paw balm paw cream 
            balm cream lotion ointment salve moisturizer skin care coat care fur care dental care 
            toothbrush toothpaste dental chew dental stick plaque remover breath freshener 
            toy toys play toy bone ball frisbee rope tug squeaky chew tug toy plush toy toy stuffed 
            collar leash harness halter lead line tie-down tether carrier crate cage pen bed bedding 
            blanket cushion mat pillow bolster pad covering lining furniture cover couch cover 
            food bowl water bowl feeding dish drinking dish feeding mat placemat tray litter box 
            litter litter pan litter scoop liner pee pad training pad absorbent pad puppy pad 
            waste bag poop bag pickup bag disposal biodegradable compostable scoop waste container 
            grooming brush comb rake dematting tool scissors clippers clipper nail trimmer nail file 
            styptic powder ear cleaner eye drops eye ointment contact solution saline solution 
            medication medicine pill tablet capsule liquid suspension powder injection antibiotic 
            antifungal anti parasitic flea tick heartworm prevention deworm dewormer probiotic 
            enzyme digestive aid prebiotic fermented food kelp seaweed omega fatty acid fish oil 
            hemp CBD cannabidiol cannabis THC calming supplement calming treat anxiety relief 
            joint support hip elbow knee supplement glucosamine chondroitin MSM turmeric 
            pain relief pain management NSAID aspirin tramadol gabapentin carprofen meloxicam 
            antihistamine allergy relief antihistamine desensitization allergy shots immunotherapy 
            vitamins minerals trace elements electrolytes hydration powder drink mix water additive 
            probiotic prebiotic fiber supplement digestive health GI health gut health microbiome 
            ingredient safety non toxic toxic harmful dangerous allergenic allergen allergentic 
            artificial ingredient artificial color artificial flavor artificial sweetener 
            preservative natural preservative BHA BHT ethoxyquin TBHQ propylene glycol BVO 
            fillers binders thickeners emulsifiers stabilizers gelling agent kelp seaweed 
            grain gluten wheat corn soy dairy egg fish shellfish tree nut peanut sesame 
            chemical ingredient safety ingredient sourcing ingredient origin quality testing 
            third party testing certification organic natural non GMO non-GMO vegan vegetarian 
            cruelty free sustainable sourcing fair trade ethical sourcing made in USA manufacturing 
            allergen risk allergen warning cross contamination allergy alert intolerance sensitivity 
            brand name manufacturer product line variant flavor size weight ounces pounds grams ml 
            price cost value cost per serving cost per unit best value budget friendly premium luxury 
            age group puppy adult senior senior plus geriatric breed size toy small medium large giant 
            breed specific formula health condition therapeutic prescription limited ingredient 
            novel protein hypoallergenic breed specific sensitive skin digestive health joint health 
            kidney health liver health cardiac heart health diabetes diabetic prescription diet 
            urinary health UTI cystitis crystal crystal prevention struvite oxalate uric acid 
            weight management weight loss low calorie restricted calorie dental health plaque tartar 
            skin coat hair shine luster quality wellness health benefits health claim health promotion 
            nutritional value nutritional benefit balanced complete nutrition aafco certified aafco 
            ingredient list ingredient breakdown ingredient sourcing ingredient quality ingredient safety 
            rating score review reviews customer review user review testimonial feedback recommendation 
            compatibility breed compatibility size compatibility age compatibility dietary compatibility 
            life stage compatibility health condition compatibility special needs special dietary 
            restrictions restricted ingredient limited ingredient hypoallergenic prescription limited diet 
            behavioral benefits training aid calming anti anxiety aggression management anxiety relief 
            stress relief relaxation comfort soothing calming behavioral support emotional support 
            probiotics bacteria beneficial bacteria microbiome gut flora digestive health gut health 
            digestibility digestible bioavailability absorption nutrient absorption nutrient retention 
            stool quality firm stool feces odor smell digestive upset GI upset diarrhea constipation 
            vomit vomiting regurgitation food sensitivity food intolerance food allergy reaction 
            ingredients sourcing sourcing practice ethical sourcing fair trade sustainable practices 
            environmental impact sustainability eco friendly recycled packaging packaging material 
            freshness expiration date shelf life storage instructions storage requirements temperature 
            recalls product recalls safety recall contamination contaminated product batch lot number 
            warning caution use as directed follow instructions dosage dose administration method 
            frequency frequency duration days weeks months years continue indefinitely temporary 
            side effects side effect adverse reaction adverse effect potential risk contraindication 
            interaction drug interaction food interaction supplement interaction medication interaction 
            vet recommended veterinary approved veterinary prescription prescription required OTC 
            over the counter without prescription without veterinary approval self administered 
            easier alternative better alternative similar product competitor product comparison 
            price comparison value comparison benefit comparison quality comparison ingredient comparison 
        '''.split())
        
        # ===== HOME ENVIRONMENT SAFETY SCAN =====
        self.home_safety_keywords = set('''
            home environment safety scan room kitchen bedroom bathroom living room balcony patio 
            yard garden garage basement laundry room closet pantry attic crawl space outdoor indoor 
            toxic plant poisonous plant dangerous plant lily lilies sago palm tulip daffodil 
            oleander foxglove azalea rhododendron ivy philodendron pothos dieffenbachia peace lily 
            snake plant jade plant kalanchoe aloe euphorbia croton yew hydrangea amaryllis daffodil 
            narcissus hyacinth muscari ivy ivy leaf toxic exposure ingestion swallowing eating 
            chemical chemicals pesticide herbicide insecticide fungicide rodenticide rat poison 
            cleaning supplies disinfectant bleach ammonia detergent soap drain cleaner oven cleaner 
            bathroom cleaner surface cleaner furniture polish wood stain paint thinner varnish 
            antifreeze coolant motor oil gasoline kerosene propane gas carbon monoxide smoke 
            medications medication pills tablets capsules liquids ointments creams patches 
            prescription drugs OTC drugs over the counter drugs ibuprofen aspirin acetaminophen 
            anti inflammatory steroid hormone birth control contraceptive estrogen testosterone 
            vitamin supplement mineral supplement herbal supplement nutraceutical recreational drug 
            illegal drug cocaine methamphetamine marijuana cannabis THC narcotic opiate heroin 
            alcohol ethanol alcoholic beverage beer wine liquor spirits 
            sharp edges sharp objects pointed objects needle pin tack thumbtack staple glass 
            broken glass shards fragments splinters sharp metal sharp plastic broken ceramic 
            razor blade razor blade holder knife kitchen knife steak knife butter knife blade 
            box cutter craft knife scissors nail screw bolt exposed screw exposed bolt rusty metal 
            rough edges rough surface burrs frayed fraying torn torn edge rough texture abrasion risk 
            pinch points pinch hazard door hinge hinge trap crush risk crushing risk squeezed squeeze 
            wire electrical wire exposed wire live wire cable cord extension cord power cord 
            outlet outlet cover outlet protection electrical hazard electrocution shock burn 
            appliance stove oven stovetop microwave toaster coffee maker blender mixer 
            heat source hot surface heat lamp heating pad hot glue gun soldering iron iron appliance 
            radiator heater furnace fireplace wood stove pellet stove heat source burn risk 
            temperature heat temperature extremes cold heat cold exposure hypothermia hyperthermia 
            water water bowl water hazard drowning risk pool pond water feature fountain waterfall 
            bathtub toilet bucket pail container standing water stagnant water contaminated water 
            small objects choking hazard choking risk swallowing risk ingestion risk blockage risk 
            marble marble size object bead coin button battery lithium battery zinc battery 
            toy parts toy stuffing toy filling beads pellets plastic pieces rubber pieces fabric 
            thread string yarn rubber band hair elastic hair tie fishing line floss dental floss 
            tinsel ribbon bow wrapping paper wrapper candy wrapper chocolate wrapper plastic wrap 
            ziplock bag plastic bag plastic sheet plastic film cling wrap saran wrap cellophane 
            rubber eraser rubber toy rubber band rubber stopper sponge foam foam pad foam block 
            insulation material loose insulation fiberglass wool mineral fiber cellulose 
            paint paint fumes lead paint lead based paint lead contamination lead exposure 
            insulation insulation material fiberglass mineral fiber asbestos harmful fiber dust 
            mold mildew moisture water damage damp damp area humidity excessive humidity basement 
            escape route escape path exit barrier fence gate door wall opening hole gap 
            fence hole gap gap under fence dig site digging area loose boards broken board 
            unstable structure weak structure wobbly unsteady collapse risk collapse structural 
            noise loud noise disturbing noise alarm siren car horn vacuum cleaner alarm system 
            stress stress trigger stressor anxiety trigger fear trigger phobia trigger reaction 
            light bright light strobe light flickering light constant light insufficient light dark 
            temperature temperature extremes hot cold cold heat fluctuation temperature swings 
            ventilation air circulation air quality indoor air quality air purifier dehumidifier 
            draft breeze wind gust air current air flow stagnant air air exchange 
            cleaning supplies storage storing chemicals secure storage locked storage child proof lock 
            pet proof storage locked cabinet locked container secure storage off limits unreachable 
            floor floor hazard floor condition slippery floor polished floor wet floor water floor 
            stairs staircase step riser stair climb descent steep stairs slippery stairs 
            carpet rug mat runner non skid non slip anti slip area rug throw rug loose rug 
            furniture furniture stability furniture placement unstable furniture wobbly furniture 
            couch sofa chair table leg table edge counter edge sharp edge corner protector 
            cushion pillow bedding fabric sofa arm ottoman chaise lounge recliner lazy boy 
            window window treatment blinds curtains drapes shades plantation shutter blind cord 
            cord strangulation strangulation hazard cord trap entanglement choking hazard window blind 
            door door slam pinch point door gap door hinge door closure door mechanism 
            trash trash can garbage can trash bin trash lid trash disposal container toxic trash 
            food scraps leftover food rotting food spoiled food mold moldy food mycotoxin 
            floor cleaning floor wax polishing wax slippery surface non skid surface anti slip 
            rug placement rug position rug stability securing rug preventing slipping 
            pathway pathway clear cluttered pathway hazard navigation mobility movement access 
            ceiling ceiling height low ceiling fan blade ceiling fixture pendant light chandelier 
            hanging object hanging decoration hanging plant hanging basket lamp shelf hanging 
            shelf bookshelf wall shelf floating shelf stability weight capacity overloaded shelf 
            picture frame artwork wall hanging mirror mirror stability mirror safety frame secure 
            plant plant pot plant soil plant water standing water water container water stagnation 
            light exposure UV exposure natural light window sun exposure sunny location shady location 
            climate temperature humidity climate control season seasonal temperature seasonal humidity 
            safety score risk level hazard level safe unsafe moderately risky highly risky 
            fix recommendation improvement suggestion modification alteration enhancement addition 
            pet proofing puppy proofing senior proofing safety proofing environment preparation 
            safety measure safety device safety equipment baby gate child safety gate pressure gate 
            crate enclosure pen playpen x pen exercise pen safe space safe room safety perimeter 
            barrier blockade partition gate fence wall door closed door separation access control 
        '''.split())
        
        # ===== INJURY ASSISTANCE =====
        self.injury_keywords = set('''
            injury injuries hurt trauma wound wounds cut cuts scrape scrapes abrasion abrasions 
            laceration lacerations tear torn rip ripped wound open wound closing wound healing 
            bite bite wound bite mark bite injury scratch scratch mark scratch wound claw mark 
            puncture puncture wound penetrating wound object penetration embedded object stuck 
            burn burns burned scorched heat burn thermal burn chemical burn acid burn alkali 
            blister blistering blistered fluid filled blister open blister ruptured blister 
            fracture broken bone break bone crack bone splinter splinter embedded object 
            sprain strain muscle strain ligament strain ligament injury tendon injury tendon tear 
            dislocation dislocated joint joint out of place subluxation luxation reduce relocation 
            bleeding bleed bleeding heavily hemorrhage hemorrhaging nose bleed nosebleed ear bleed 
            blood loss blood oozing blood trickling blood dripping bloodstain bleeding control 
            clotting coagulation blood clot blood stopped clotting time bleeding time tourniquets 
            bruise bruising bruised contusion ecchymosis black and blue discoloration purple swelling 
            swelling swollen puffed inflamed inflammation edema fluid accumulation increased size 
            paw injury paw pad injury paw laceration paw abrasion pad injury pad wound pad cut 
            paw pad crack paw pad bleed nail bed injury nail injury nail bleed torn nail split nail 
            claw claw injury claw split claw crack claw overgrown claw retraction retracted claw 
            nose injury nasal injury nose bleed epistaxis nasal trauma nasal fracture nasal bleeding 
            ear injury ear trauma ear bleed ear laceration ear flap ear hematoma aural hematoma 
            eye injury eye trauma eye laceration eye abrasion corneal abrasion corneal ulcer 
            eye opening eye closing eyelid injury eyelid laceration eyelid swelling conjunctival 
            mouth injury mouth trauma oral trauma mouth bleed mouth laceration gum injury tooth injury 
            throat injury throat laceration esophageal injury esophageal trauma tracheal injury 
            chest injury chest trauma chest bleed chest wall injury broken rib rib fracture flail rib 
            abdominal injury abdominal trauma abdominal bleed internal bleeding organ injury 
            internal bleeding internal injury hemorrhage internal hemorrhage shock hypovolemic 
            limb injury limb laceration limb fracture limb bleed limb swelling lameness limping 
            joint injury joint swelling joint fluid joint effusion knee injury hip injury elbow injury 
            tail injury tail fracture tail bleed tail swelling tail bend kink 
            genital injury genital trauma genital bleed testicular injury penile injury vulvar injury 
            rectal injury anal injury anal tear anal trauma rectal tear rectal bleeding 
            severity minor mild moderate severe critical life threatening emergency urgent 
            blood amount small amount large amount profuse bleeding heavy bleeding light bleeding 
            oozing trickling dripping flowing spurting gushing arterial bleeding venous bleeding 
            wound depth surface wound superficial shallow wound deep wound penetrating wound 
            wound appearance clean wound dirty wound contaminated wound foreign material debris 
            infection risk infection prevention infected wound signs infection inflammation 
            fever chills malaise lethargy weakness swelling heat warmth redness pain pus 
            pain level pain intensity mild pain moderate pain severe pain excruciating pain 
            pain response guarding tenseness vocalization yelping crying whimpering limping 
            first aid immediate care emergency treatment basic treatment wound care 
            cleaning cleaning steps washing rinsing flushing saline solution hydrogen peroxide 
            betadine iodine chlorhexidine alcohol isopropyl alcohol sterile sterile gauze 
            bandaging bandage band aid gauze pad tape medical tape ace bandage elastic bandage 
            compression dressing pressure dressing pressure wrap elevation elevation technique 
            ice pack ice cold compress cold therapy cryotherapy pain relief anesthesia pain control 
            hemorrhage control bleeding control direct pressure tourniquet pressure bandage 
            immobilization immobilize splint rigid splint sling collar cone Elizabethan collar 
            pain medication pain reliever analgesic NSAID anti inflammatory pain management 
            topical cream topical ointment topical treatment antibiotic ointment antiseptic ointment 
            wound dressing dressing type dressing change dressing removal gauge gauze packing 
            suture suturing stitches stitching closure tape skin adhesive glue medical glue 
            steri strip butterfly bandage adhesive strip steri strip skin closure method 
            infection prevention antiseptic cleaning antibiotic ointment topical antibiotic 
            tetanus tetanus risk tetanus prevention tetanus shot booster vaccination 
            rabies rabies risk rabies concern rabies prevention rabies post exposure prophylaxis 
            follow up follow up care wound check recheck examination monitoring observation 
            healing timeline healing process healing rate expected healing scarring scar appearance 
            vet visit emergency vet urgent care veterinary treatment professional care consultation 
            when to seek help when to call vet red flags warning signs concerning signs 
            home treatment home care DIY self care self treatment over the counter OTC OTC treatment 
            emergency emergency situation life threatening threatening circulation loss consciousness 
            breathing difficulty respiratory distress shock signs shock treatment shock prevention 
            CPR cardiopulmonary resuscitation rescue breathing mouth to mouth artificial respiration 
            tools supplies bandages gauze tape scissors antiseptic ointment pain medication 
            preparation preparation steps planning ahead emergency kit first aid kit emergency supplies 
            OTC supplies OTC products over the counter products topical treatments topical ointments 
            supplies needed tools needed what to have what to get what to buy first aid kit contents 
            wound assessment assess wound examine wound look at wound evaluate wound inspect wound 
            debris removal remove debris clean wound foreign material extraction 
            bleeding assessment bleeding control hemorrhage assessment assess bleeding 
            shock assessment assess shock monitor shock signs shock symptoms treatment 
        '''.split())
        
        # ===== POOP & VOMIT ANALYSIS =====
        self.poop_vomit_keywords = set('''
            poop feces stool bowel movement defecation poo stool consistency fecal consistency 
            vomit vomitus vomiting regurgitation throwing up retch retching heaving nausea queasy 
            digestive issue gastrointestinal GI upset digestive upset stomach upset gut upset 
            blood blood in stool blood in poop bloody stool bloody poop hematochezia 
            bright red blood darker blood melena tarry stool black stool coffee ground 
            blood in vomit bloody vomit hematemesis blood tinged vomit 
            mucus mucus in stool slimy stool stringy stool mucoid stool jelly-like stool 
            mucus in vomit mucoid vomit stringy vomit foamy vomit frothy vomit 
            worm worms parasite parasites visible parasite worm segment roundworm hookworm tapeworm 
            white worm rice grains rice-like segments flat worm round worm thread-like worm 
            undigested food food particles kibble kibble pieces food pieces incomplete digestion 
            color pale stool dark stool black stool tarry stool clay colored stool gray stool 
            yellow stool orange stool green stool brown stool red stool bloody stool 
            texture consistency firmed stool formed stool well formed stool loose stool 
            mushy stool pasty stool diarrhea watery diarrhea liquid stool soft stool 
            diarrhea diarrheal loose stool soft stool mushy stool watery watery stool liquid 
            constipation constipated hard stool difficult defecation strain straining difficulty 
            volume large volume small volume excessive diarrhea multiple stools frequent defecation 
            frequency defecation frequency bowel movement frequency times per day daily frequency 
            urgency urgent defecation sudden urge sudden onset straining defecation difficult 
            odor foul smell strong smell pungent odor odorous stool foul smelling stool 
            gas flatulence bloating abdominal bloating abdominal distension bloated belly 
            appetite appetite loss decreased appetite anorexia increased appetite polyphagia 
            eating eating habits food intake meal intake snacking between meals frequent feeding 
            water intake water consumption drinking habits hydration dehydration over hydration 
            thirst excessive thirst polydipsia decreased thirst water intake normal water intake 
            weight weight loss weight gain stable weight weight fluctuation rapid weight loss 
            lethargy lethargy sluggish slow inactive inactive sluggish tired fatigue exhausted 
            energy level energy decreased energy low energy high energy hyperactive normal energy 
            behavior behavioral change change in behavior normal behavior abnormal behavior 
            posture posture changes hunched posture pain posture tension stretched posture 
            abdominal pain abdominal discomfort belly pain abdominal tenderness guarding tenseness 
            recent changes recent illness recent food change diet change food change transitions 
            stress stress induced diarrhea stress-related digestive upset anxiety stress trigger 
            medication medication change medication side effect medication history recent medication 
            medical history medical conditions previous illness illness history recurring problem 
            food diet food type diet change new food novel diet food allergy food sensitivity 
            food intolerance dietary indiscretion eating inappropriate items eating garbage 
            toxic ingestion poisoning toxicity toxic exposure chemical exposure contamination 
            medications medication ingestion accidental ingestion toxin toxins toxicity poison 
            foreign object foreign body object ingestion obstruction blockage impaction 
            dehydration rehydration hydration status fluid status fluid loss fluid intake 
            electrolyte electrolyte imbalance mineral imbalance sodium potassium calcium magnesium 
            infection viral infection bacterial infection parasitic infection protozoan infection 
            bacteria bacterial overgrowth dysbiosis microbiome imbalance gut flora imbalance 
            probiotic probiotic supplementation probiotic therapy prebiotic fermented food 
            enzyme enzyme supplementation digestive enzyme pancreatic enzyme carbohydrate protease 
            pancreas pancreatic insufficiency pancreatitis pancreatic disease pancreatic function 
            liver liver disease liver dysfunction hepatic disease bile flow bilirubin liver enzyme 
            kidney kidney disease kidney dysfunction renal disease uremia kidney function 
            intestine intestinal inflammation intestinal disease inflammatory bowel disease IBD 
            colon colitis ulcerative colitis colonic inflammation colonic disease colon disease 
            small intestine small bowel enteritis duodenitis jejunitis ileitis small bowel disease 
            stomach gastric stomach inflammation gastritis gastric upset gastric disease stomach pain 
            esophagus esophageal inflammation esophagitis esophageal disease esophageal reflux 
            GERD gastric reflux acid reflux regurgitation reflux disease 
            viral disease viral infection rotavirus parvovirus coronavirus distemper panleukopenia 
            bacterial disease bacterial infection salmonella E coli Clostridium campylobacter 
            parasitic disease parasitic infection giardia coccidia cryptosporidium hookworm roundworm 
            protozoan protozoal disease protozoan infection giardia coccidia trichomonas 
            worm burden parasite load parasites present parasite medication antiparasitic dewormer 
            toxin toxins mycotoxin aflatoxin mold toxin plant toxin food toxin chemical toxin 
            poison mushroom poison toxin plant poison chemical poison accidental poisoning 
            toxicity signs toxicity symptoms toxicity effects 
            home treatment home management supportive care dietary management diet modification 
            bland diet bland food chicken rice boiled diet easily digestible diet 
            feeding schedule feeding frequency meal frequency meal timing feeding intervals 
            hydration hydration support fluid support water access drinking water 
            recovery recovery time recovery process healing healing timeline expected recovery 
            monitoring monitoring observations monitoring symptoms monitoring changes 
            vet evaluation vet assessment veterinary examination professional evaluation consultation 
            fecal test fecal analysis fecal flotation parasite testing stool sample 
            culture sensitivity culture test bacterial culture antibiotic sensitivity 
            endoscopy endoscopic examination colonoscopy gastroscopy esophagoscopy 
            ultrasound ultrasound examination abdominal ultrasound imaging 
            radiographs X-rays radiographic imaging abdominal radiographs 
            blood test blood work chemistry panel CBC complete blood count 
            stool character stool appearance stool quality assessment evaluation 
            diagnostic testing diagnostic evaluation diagnostic workup diagnosis confirmation 
        '''.split())
        
        # ===== PARASITE DETECTION =====
        self.parasite_keywords = set('''
            parasite parasites parasitic infection external parasite internal parasite 
            flea fleas flea infestation flea bite flea allergy dermatitis flea control 
            flea prevention flea medication flea treatment flea collar flea comb flea shampoo 
            flea lifecycle flea eggs flea larvae flea pupae flea adult fleas on pet 
            tick ticks tick infestation tick bite tick-borne disease tick removal 
            tick prevention tick medication tick treatment tick collar tick repellent 
            mite mites mite infestation mange demodex sarcoptes otodectic ear mite 
            mange sarcoptic mange demodectic mange mange treatment mange medication 
            louse lice lice infestation pediculosis louse removal lice shampoo lice comb 
            visible parasite parasite visible parasite crawling parasite moving visible 
            skin lesion skin irritation skin damage skin infection skin inflammation 
            itching itchy scratching excessive scratching self trauma self mutilation 
            hair loss alopecia hair loss excessive shedding bald patches bald spots 
            red skin redness inflamed skin skin irritation dermatitis contact dermatitis 
            scabs scabbing crusty skin crusty lesion open sore sore wound 
            dandruff dry skin flaky skin scaling seborrhea seborrheic dermatitis 
            ear discharge ear debris ear mite debris dark debris waxy discharge 
            ear infection ear inflammation otitis externa otitis media ear pain 
            head shaking head tilting scratching ears ear sensitivity ear pain 
            anal sac anal glands scooting dragging butt anal itching perianal itching 
            anal area irritation perianal dermatitis anal gland impaction anal gland disease 
            stool appearance stool changes diarrhea constipation blood in stool mucus in stool 
            vomit vomiting regurgitation digestive upset abdominal pain 
            lethargy sluggish inactive behavior change weight loss decreased appetite 
            roundworm roundworms nematode ascarid ascaris toxascaris toxocara hookworm 
            tapeworm tapeworms cestode dipylidium segmented worm rice grains rice-like 
            whipworm trichuris threadworm strongyloides 
            giardia giardiasis giardia cysts giardia trophozoites giardia infection 
            coccidia coccidiosis coccidia oocysts eimeria isospora 
            cryptosporidium cryptosporidiosis 
            trichomonas trichomoniasis 
            heartworm dirofilaria heartworm disease heartworm infection microfilaremia 
            cough exercise intolerance heart disease cardiac disease fatigue 
            heartworm prevention heartworm medication heartworm treatment 
            mosquito mosquitoes heartworm transmission mosquito bite prevention 
            lungworm angiostrongylus cough respiratory signs 
            blood parasite babesia bartonella ehrlichia rickettsial disease tick-borne 
            skin scraping skin scrape microscopic examination parasite identification 
            fecal examination fecal test fecal flotation fecal smear stool sample 
            blood test blood smear microfilariae circulating microfilariae 
            parasite lifecycle parasite transmission parasite control prevention 
            parasite medication antiparasitic anthelmintic anthelmintic treatment 
            topical treatment spot on flea tick medication pour-on treatment 
            oral medication oral tablet oral liquid oral suspension antiparasitic oral 
            combination treatment flea tick heartworm combination all-in-one 
            parasite burden parasite load heavy infestation light infestation moderate 
            treatment protocol treatment plan treatment schedule treatment duration 
            recheck re-treatment follow-up follow up treatment follow-up testing 
            environmental treatment environmental control home treatment 
            washing bathing bathing protocol shampoo antiparasitic shampoo 
            decontamination decontamination process cleaning protocol sanitization 
            bedding cleaning bedding washing bedding replacement 
            yard treatment yard spraying yard treatment pest control 
            fecal disposal proper disposal sanitation hygiene 
            concurrent parasites multiple parasites co-infection dual infection 
            seasonal parasite seasonal treatment year-round prevention 
            indoor outdoor risk indoor risk outdoor risk exposure risk 
            vaccination vaccination parasite prevention immunization 
            resistance drug resistance parasite resistance treatment failure 
            side effects adverse reactions medication side effects reactions 
            contraindication contraindications medication interactions drug interactions 
            age appropriateness age appropriate dosing weight-based dosing 
            immunity immune response protective immunity acquired immunity 
            transmission risk transmission route transmission prevention 
            zoonotic zoonotic parasite zoonotic risk human infection pet to human 
            handling precautions safety precautions hygiene precautions hand washing 
            pregnancy pregnant dog pregnant cat pregnancy contraindication 
            nursing nursing dam lactation nursing considerations 
            toxicity toxicity risk toxicity symptoms side effect profile 
        '''.split())
        
        # ===== EMOTION ANALYSIS VIA BODY LANGUAGE =====
        self.emotion_keywords = set('''
            emotion emotions emotional state emotional behavior body language behavior 
            happy happiness joy excited playful playfulness fun enjoying wagging tail 
            tail wagging fast tail wagging slow tail wagging height tail position 
            ears ears forward ears alert ears upright ear position ear movement 
            mouth mouth open smile pant open-mouthed tongue out tongue lolling 
            eyes bright alert eyes soft eyes open wide eyes normal 
            body posture stance posture forward lean play bow play position 
            bouncing bouncy bounding energetic dynamic movement 
            engagement engaging interactive social seeking attention attention seeking 
            playfulness playful play biting play wrestle play stance play bow 
            sad sadness depressed depression down mood sad mood melancholy 
            tail down tail tucked tail between legs tail low tail position 
            ears back ears drooping ears pinned back ear position 
            eyes dull eyes closed eyes withdrawn eyes sad puppy dog eyes 
            posture withdrawn withdrawn posture slouched posture low posture collapsed 
            stillness still motionless lack movement static inactive 
            isolation isolated withdrawn withdrawing antisocial avoiding 
            lethargy sluggish inactive slow moving lack energy lethargy 
            appetite loss lack interest disinterest lack motivation 
            fear fearful afraid scared frightened anxious nervous worried 
            ears back ears pinned ears flat against head ear position 
            tail tucked tail between legs tail low cringing 
            body posture crouched posture lowered posture cowering ducking 
            trembling shaking trembling quivering nervous shaking 
            eyes wide wide-eyed pupils dilated eyes showing whites dilated pupils 
            mouth closed mouth tense tight mouth closed mouth pulled back 
            hackles hackles raised piloerection hair standing on end 
            freezing frozen motionless still statue-like 
            avoidance avoiding hiding hiding behavior avoidant 
            submission submissive submissive posture deferential 
            play solicitation play request seeking interaction initiating play 
            anxiety anxious worried nervous stressed tension 
            tension tense muscles tight tight jaw tension in body 
            panting panting heavily rapid breathing rapid respiration 
            whining whining vocalizing vocalization soft vocalization 
            stress stressed tension stressed-out overwhelmed 
            stress response stress behavior stress sign stress indicator 
            aggression aggressive aggressive behavior aggressive stance 
            growling growl growling vocalization aggressive vocalization 
            snarling snarl snarling bared teeth showing teeth 
            teeth bared teeth showing canines exposed teeth visible 
            lunging lunging forward lunge attack posture threatening posture 
            snapping snapping at biting snapping biting 
            ears back ears pinned ears tense ear position 
            eyes hard eyes intense stare intense eyes focused 
            body posture stiff posture rigid posture forward posture 
            forward lean forward lean aggressive stance 
            barrier motion pacing pacing behavior restless restlessness 
            hackles raised hackles piloerection hair standing on end 
            territorial territorial marking territorial behavior territory guarding 
            dominance dominant dominance display dominance gesture 
            calm calmness calm demeanor calm peaceful relaxed 
            soft eyes soft gentle eyes relaxed eyes normal eyes 
            ears neutral ears forward ears relaxed ear position 
            mouth closed mouth relaxed loose mouth natural mouth position 
            body posture relaxed posture loose posture soft posture 
            tail neutral tail slightly wagging neutral tail position 
            movement smooth smooth movement fluid movement natural movement 
            stillness still rest resting settling quiet resting 
            breathing normal breathing regular breathing calm breathing 
            engaged engaged interaction responsive social interactive 
            confidence confident assured self-confident bold 
            shoulders back shoulders relaxed shoulder position 
            head up head position natural head position forward head 
            curiosity curious inquisitive interested interesting 
            exploration exploring investigating checking out sniffing 
            friendly friendly behavior friendly demeanor welcoming open 
            greeting greeting behavior hello greeting posture 
            approach approaching coming forward social approaching 
            frustration frustrated frustrated behavior frustration signs 
            jealousy jealous envious possessive possessiveness 
            possessive possessiveness guarding guarding behavior resource guarding 
            obsessive obsessive behavior obsessive focus repetitive 
            compulsive compulsive behavior compulsive movements repetitive movements 
            OCD obsessive-compulsive disorder obsessive compulsive 
            attention-seeking attention seeking demanding demanding behavior 
            clingy clingy behavior separation anxiety abandonment anxiety 
            independence independent aloof aloof behavior stand-offish 
            indifference indifferent apathy apathetic unmotivated 
            confusion confused confusing confused behavior disoriented 
            disorientation disorientated loss of awareness 
            pain pain response pain behavior guarding behavior protective behavior 
            nociception nociceptive pain response pain indication 
            sensitivity sensitive touch sensitive tender sensitive area 
            body signal body signals pain signals behavioral signals 
            training training behavioral training behavior modification training response 
            motivation motivated highly motivated low motivation unmotivated 
            drive prey drive play drive food drive toy drive 
            reactivity reactive over-reactive under-reactive responsiveness 
            startle response startle easily being easily startled jump reaction 
            confidence lack of confidence insecurity self-doubt shy timid 
            boldness bold fearless confident assertive 
            social interaction social skills social engagement socialization 
            solitary solitary behavior loner preference alone preference 
            pack behavior pack dynamics hierarchy dominance submission 
            cooperation cooperative cooperative behavior willing obedient 
            resistance resistant defiant stubborn noncompliant resistant behavior 
            tolerance pain tolerance stress tolerance frustration tolerance 
            resilience resilient bouncing back quick recovery emotional stability 
            sensitivity emotional sensitivity sensitive sensitive to emotion 
            bonding bond bonded attachment attachment secure attachment insecure 
            trust trusting trustworthy trusted relationship 
            distrust mistrustful wary cautious suspicious 
            connection connected strong connection weak connection disconnected 
            responsiveness responsive reactive responsive to stimuli 
            assessment assessment body language assessment emotional assessment 
            interpretation interpretation body language interpretation emotion detection 
            communication communication via body language nonverbal communication 
            signals signals body language signals behavioral signals 
            state assessment state evaluation current state emotional state 
            change change in behavior behavioral change emotional change mood change 
            trigger trigger identification triggering factor stress trigger 
            reaction reaction behavior reaction emotional reaction stress reaction 
            coping coping mechanism coping strategy stress coping 
            management behavior management emotional management stress management 
            training training for behavior training for emotion training positive 
            reinforcement positive reinforcement reward-based training reward motivation 
            improvement improvement in behavior emotional improvement behavior change 
            support emotional support behavioral support confidence building 
            consistency consistency in training consistent behavior consistent routine 
            predictability predictable behavior predictable environment 
            environment environmental factors environmental triggers environmental stress 
            context contextual understanding situational understanding context-dependent 
        '''.split())
        
        # ===== TOY & PRODUCT SAFETY (EXPANDED) =====
        self.toy_keywords = set('''
            toy play fetch chew ball frisbee tug rope squeaky bone treat reward fun exercise 
            training obedience agility socialization companionship toy bone ball frisbee rope 
            plush squeaky stuffed animal puzzle scratcher wand tunnel rubber plastic nylon latex 
            fabric wood antler rawhide silicone splinter sharp edges swallow choke stuck blockage 
            pieces stuffing break snap ingest digestive teeth gums chew destroy shred rip play 
            fetch tug durability brand make model material non-toxic bpa-free heavy duty 
            indestructible durable longevity lifespan wear resistant tear resistant 
            toy quality quality construction manufacturing durable construct reinforced stitching 
            toy testing testing protocol safety testing impact testing durability testing 
            size appropriateness size pet size toy size small toy large toy toy sizing 
            age appropriate age puppy puppy toy age senior senior dog toy age adult adult dog toy 
            developmental stage teething teething toy teething pain relief chewing needs 
            play style play preference aggressive chewer moderate chewer gentle chewer 
            dog type small dog large dog giant breed toy breed toy preference 
            interest interest level toy interest engagement engagement level toy engagement 
            motivation food motivated toy motivated play motivated prey drive motivated 
            supervision supervised play unsupervised play with supervision without supervision 
            interactive toy interactive play human interaction toy-based interaction 
            solitary toy solitary play independent play self-entertainment self play 
            social play social toy multi-dog play group play 
            enrichment enrichment toy mental enrichment cognitive enrichment puzzle toy 
            stimulation mental stimulation physical stimulation sensory stimulation 
            entertainment entertainment value boredom relief boredom prevention 
            exercise exercise toy exercise promotion activity activity level 
            bonding bonding toy bonding activity strengthening relationship 
            reward toy reward training reward motivation training aid 
            rubber toy rubber durability latex allergy latex sensitivity latex-free 
            plastic toy plastic safety plastic durability BPA-free plastic non-toxic plastic 
            fabric toy fabric durability fabric quality stitching seam integrity 
            stuffing toy stuffing foam filling polyfill loose stuffing 
            squeaker squeaky toy noise level squeaker removal noise sensitivity 
            crinkle crinkly toy crinkle sound noise crinkle material 
            rope toy rope tug rope fraying rope integrity rope knots 
            ball ball durability ball size ball material bouncy ball foam ball 
            tennis ball tennis ball cover tennis ball lint tennis ball wear 
            kong kong toy kong durability rubber toy rubber durability 
            nylabone nylon chew chew toy nylon durability splinter risk 
            rawhide rawhide chew rawhide safety rawhide digestion rawhide choking 
            antler antler chew antler durability antler splintering antler safety 
            bully stick bully stick digestibility bully stick safety bully stick choking 
            dental toy dental chew plaque reduction tartar reduction cleaning toy 
            treat-dispensing treat dispenser puzzle feeder interactive puzzle 
            tug toy tug game tug-of-war rope tug rope toy tug rope 
            fetch toy fetch ball fetch stick fetch toy material fetch toy durability 
            frisbee frisbee catch frisbee flying disc flying disc material 
            tunnel tunnel toy play tunnel agility tunnel tunneling behavior 
            wand toy feather wand wand toy interactive play wand cat toy 
            mouse toy mouse prey drive mouse squeaker mouse material 
            catnip catnip toy catnip response catnip sensitivity fresh catnip dried catnip 
            feather toy feather material feather durability feather safety natural feather 
            scratch scratch toy scratching post scratching pad scratching behavior 
            scratcher scratching surface scratching preference scratching material 
            sisal sisal post sisal rope sisal scratching sisal durability 
            cardboard cardboard scratching cardboard toy cardboard scratcher 
            carpet carpet scratcher carpet post carpet scratching friction 
            climbing climbing toy climbing cat climbing structure multi-level toy 
            perch perching cat perch elevated perch window perch 
            hideaway hideaway toy hiding place hiding behavior cat cubby 
            platform platform cat platform elevated elevated surface high place 
            hanging toy hanging play hanging feather hanging toy motion 
            bell bell toy sound bell noisy toy jingle toy 
            mirror mirror toy reflective toy reflective surface cat mirror 
            outdoor toy outdoor durability outdoor material weather-resistant uv resistant 
            water toy water-resistant waterproof floating toy pool toy 
            winter toy winter durability cold weather resistant 
            summer toy summer heat resistant sun exposure uv exposure 
            battery battery toy electronic toy electronic toy safety battery safety 
            cord cord safety hanging cord entanglement cord strangulation 
            safety inspection regular inspection toy wear inspection damage inspection 
            wear and tear wear pattern damage pattern deterioration inspection 
            cleanliness clean toy sanitation toy washing regular cleaning 
            sanitization sanitizing toy sanitized deep cleaning 
            storage storage safety proper storage toy storage rotation 
            rotation toy rotation toy variation varied toys multiple toys 
            disposal disposal safe disposal appropriate disposal recycling 
            replacement toy replacement when to replace frequent replacement 
            recall recall information recalled toy product recall 
            alternative alternative toy substitute toy safer alternative 
            comparison toy comparison brand comparison quality comparison 
            recommendation toy recommendation vet recommended expert recommended 
            review customer review user review rating ratings testimonial 
            price cost value budget-friendly premium expensive 
            availability availability in stock in stock finding toy 
        '''.split())
        
        
    def extract_keywords(self,query):
        query = query.lower()
        tokens = word_tokenize(query)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        # Combine all keyword sets for extraction
        all_keywords = (
            self.diseases_keywords |
            self.health_scan_keywords |
            self.weight_keywords |
            self.food_keywords |
            self.packaged_product_keywords |
            self.home_safety_keywords |
            self.injury_keywords |
            self.poop_vomit_keywords |
            self.parasite_keywords |
            self.emotion_keywords |
            self.toy_keywords
        )
        keywords = [token for token in lemmatized_tokens if token in all_keywords]
        return keywords

    def select_strategy(self, query):
        keywords = self.extract_keywords(query)
        if len(keywords) == 0:
            return self.default_strategy
        else:
            if any(keyword in keywords for keyword in self.diseases_keywords):
                return "skin-and-health-diagnostic"
            elif any(keyword in keywords for keyword in self.food_keywords):
                return "pet-food-image-analysis"
            elif any(keyword in keywords for keyword in self.packaged_product_keywords):
                return "packaged-product-scanner"
            elif any(keyword in keywords for keyword in self.toy_keywords):
                return "toys-safety-detection"
            elif any(keyword in keywords for keyword in self.home_safety_keywords):
                return "home-environment-safety-scan"
            elif any(keyword in keywords for keyword in self.injury_keywords):
                return "injury-assistance"
            elif any(keyword in keywords for keyword in self.poop_vomit_keywords):
                return "poop-vomit-detection"
            elif any(keyword in keywords for keyword in self.parasite_keywords):
                return "parasite-detection"
            elif any(keyword in keywords for keyword in self.emotion_keywords):
                return "emotion-detection"
            else:
                return self.default_strategy
