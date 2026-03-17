"""Semantic-gap dataset for query–document alignment research.

Design principles
-----------------
Every (query, positive_doc) pair is **semantically related but lexically
divergent** — they share meaning but NOT surface tokens.  This forces an
embedding model to capture deep semantics rather than keyword overlap.

Gap categories covered
~~~~~~~~~~~~~~~~~~~~~~
1. Synonym / paraphrase          (cheap → budget-friendly)
2. Colloquial → professional     (get rid of → eliminate)
3. Abstract → concrete           (improve health → balanced diet plan)
4. Cross-lingual hint            (Chinese query → English doc, mixed)
5. Intent re-framing             (question form → declarative answer)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Triplet:
    """A single (query, positive_doc, negative_doc) sample."""
    query: str
    positive_doc: str
    negative_doc: str
    gap_type: str  # for analysis / stratified eval


def load_semantic_gap_dataset() -> List[Triplet]:
    """Return the curated semantic-gap triplet dataset.

    Returns
    -------
    list[Triplet]
        30 hand-crafted triplets spanning multiple gap categories.
    """
    return [
        # ── 1. Synonym / paraphrase ──────────────────────────────
        Triplet(
            query="cheap flights to Tokyo",
            positive_doc="Budget-friendly airfare options for travel to Tokyo Narita and Haneda airports, including discount carriers and off-peak seasonal pricing strategies.",
            negative_doc="Luxury first-class cabin reviews for transatlantic routes between New York and London Heathrow.",
            gap_type="synonym",
        ),
        Triplet(
            query="how to get rid of a headache fast",
            positive_doc="Rapid migraine relief techniques: cold compress application, hydration protocols, and over-the-counter analgesic dosing guidelines for acute cephalgia.",
            negative_doc="Comprehensive guide to building a home gym on a budget with resistance bands and adjustable dumbbells.",
            gap_type="synonym",
        ),
        Triplet(
            query="easy way to lose weight",
            positive_doc="Evidence-based body composition management through caloric deficit strategies, sustainable nutritional planning, and moderate-intensity aerobic exercise regimens.",
            negative_doc="Advanced deadlift programming for competitive powerlifters preparing for national-level meets.",
            gap_type="synonym",
        ),
        Triplet(
            query="best phone with good camera",
            positive_doc="Flagship smartphone computational photography comparison: sensor size, aperture, night mode algorithms, and multi-lens optical zoom capabilities ranked for 2025.",
            negative_doc="Enterprise network switch configuration guide for Cisco Catalyst 9000 series in data center environments.",
            gap_type="synonym",
        ),
        Triplet(
            query="fix slow computer",
            positive_doc="System performance optimisation procedures: disk defragmentation, startup program management, RAM utilisation diagnostics, and malware scan remediation workflows.",
            negative_doc="Introduction to watercolour painting techniques for beginners including wet-on-wet methods.",
            gap_type="synonym",
        ),

        # ── 2. Colloquial → professional ─────────────────────────
        Triplet(
            query="my knee hurts when I run",
            positive_doc="Patellofemoral pain syndrome: etiology, biomechanical assessment of lower-extremity kinetic chain dysfunction, and evidence-based rehabilitation protocols for runners.",
            negative_doc="Annual maintenance schedule for Honda Civic 2024 including oil change intervals and tire rotation.",
            gap_type="colloquial_to_professional",
        ),
        Triplet(
            query="why does my car make a weird noise when turning",
            positive_doc="Differential diagnosis of steering-associated vehicular acoustics: CV joint wear, power steering fluid cavitation, and wheel bearing degradation symptomatology.",
            negative_doc="Recipe for traditional Italian tiramisu with mascarpone cream and espresso-soaked ladyfingers.",
            gap_type="colloquial_to_professional",
        ),
        Triplet(
            query="my dog keeps scratching itself",
            positive_doc="Canine pruritus differential diagnosis: atopic dermatitis, flea allergy dermatitis, sarcoptic mange, and secondary bacterial pyoderma assessment and treatment.",
            negative_doc="Beginner's guide to indoor succulent gardening with recommended species for low-light environments.",
            gap_type="colloquial_to_professional",
        ),
        Triplet(
            query="baby won't stop crying at night",
            positive_doc="Neonatal nocturnal distress management: colic identification, sleep regression phases, circadian rhythm entrainment strategies, and graduated extinction sleep training methodology.",
            negative_doc="Stock market technical analysis: understanding Bollinger Bands and moving average convergence divergence indicators.",
            gap_type="colloquial_to_professional",
        ),
        Triplet(
            query="feeling tired all the time",
            positive_doc="Chronic fatigue differential workup: thyroid function panel, iron studies and ferritin levels, vitamin D assay, sleep architecture polysomnography, and adrenal cortisol evaluation.",
            negative_doc="History of the Byzantine Empire from the fall of Rome to the Ottoman conquest of Constantinople.",
            gap_type="colloquial_to_professional",
        ),

        # ── 3. Abstract → concrete ───────────────────────────────
        Triplet(
            query="improve my health",
            positive_doc="A 12-week structured wellness programme consisting of 150 minutes weekly moderate-intensity cardio, Mediterranean dietary pattern adoption, 7-9 hour sleep hygiene, and mindfulness-based stress reduction.",
            negative_doc="Detailed comparison of PostgreSQL and MySQL query optimiser strategies for high-throughput OLTP workloads.",
            gap_type="abstract_to_concrete",
        ),
        Triplet(
            query="learn programming",
            positive_doc="Structured software engineering curriculum: Python fundamentals via interactive exercises, data structures with LeetCode progression, Git version control workflows, and capstone REST API project.",
            negative_doc="Comprehensive field guide to birdwatching in the Pacific Northwest including seasonal migration patterns.",
            gap_type="abstract_to_concrete",
        ),
        Triplet(
            query="save more money",
            positive_doc="Personal finance automation framework: 50-30-20 budgeting implementation, high-yield savings account setup, automated bi-weekly transfers, and discretionary spending envelope system.",
            negative_doc="Detailed guide to pruning rose bushes in temperate climates with seasonal timing recommendations.",
            gap_type="abstract_to_concrete",
        ),
        Triplet(
            query="be more productive",
            positive_doc="Time management system combining Pomodoro technique with Eisenhower priority matrix, calendar time-blocking, weekly review rituals, and digital distraction elimination via app-level controls.",
            negative_doc="Geological formation processes of stalactites and stalagmites in limestone cave systems.",
            gap_type="abstract_to_concrete",
        ),
        Triplet(
            query="make my website faster",
            positive_doc="Web performance optimisation checklist: asset minification and Brotli compression, lazy-loaded images with AVIF format, CDN edge caching, critical CSS inlining, and Core Web Vitals LCP/FID/CLS tuning.",
            negative_doc="Beginner's guide to fermenting kombucha at home with SCOBY culture maintenance tips.",
            gap_type="abstract_to_concrete",
        ),

        # ── 4. Cross-lingual / mixed ─────────────────────────────
        Triplet(
            query="怎么提高英语口语",
            positive_doc="Effective spoken English fluency development: shadowing technique with native speaker podcasts, spaced-repetition vocabulary systems, weekly conversation exchange partnerships, and pronunciation drills targeting connected speech patterns.",
            negative_doc="Guide complet pour préparer un coq au vin traditionnel avec des techniques de cuisine classique française.",
            gap_type="cross_lingual",
        ),
        Triplet(
            query="推荐一个好用的笔记软件",
            positive_doc="Comparative review of knowledge management tools: Obsidian's local-first markdown vault with bidirectional linking, Notion's database-driven workspace, and Logseq's outliner approach for Zettelkasten workflows.",
            negative_doc="Detailed instructions for assembling IKEA KALLAX shelf unit including required tools and estimated time.",
            gap_type="cross_lingual",
        ),
        Triplet(
            query="Python 爬虫怎么写",
            positive_doc="Web scraping with Python: requests and BeautifulSoup for static content extraction, Selenium WebDriver for JavaScript-rendered pages, rate-limiting with exponential backoff, and structured data output to JSON/CSV pipelines.",
            negative_doc="Overview of Renaissance art movements in Florence with focus on Brunelleschi's architectural innovations.",
            gap_type="cross_lingual",
        ),
        Triplet(
            query="机器学习入门看什么",
            positive_doc="Curated machine learning onboarding path: Andrew Ng's Stanford CS229 lecture series, hands-on scikit-learn tutorials, Kaggle competition progression from Titanic to tabular playground, and foundational linear algebra review via 3Blue1Brown.",
            negative_doc="Complete rules and scoring system for international cricket test matches including DLS method explanation.",
            gap_type="cross_lingual",
        ),
        Triplet(
            query="日本旅行签证怎么办",
            positive_doc="Japanese tourist visa application procedure: required documents including bank statements and employment verification, embassy appointment booking, processing timelines, and single-entry versus multiple-entry eligibility criteria.",
            negative_doc="Comprehensive maintenance guide for saltwater reef aquariums including coral placement and water chemistry.",
            gap_type="cross_lingual",
        ),

        # ── 5. Intent re-framing ─────────────────────────────────
        Triplet(
            query="is Python good for data science",
            positive_doc="Python dominates the data science ecosystem through libraries such as pandas for tabular manipulation, NumPy for numerical computation, scikit-learn for classical ML, and PyTorch/TensorFlow for deep learning pipelines.",
            negative_doc="Comprehensive Java Spring Boot microservice deployment patterns with Kubernetes orchestration.",
            gap_type="intent_reframe",
        ),
        Triplet(
            query="should I use SSD or HDD",
            positive_doc="Solid-state drives deliver 50x sequential read throughput and sub-millisecond random access latency compared to spinning disks, while HDDs retain cost-per-terabyte advantage for cold archival storage workloads.",
            negative_doc="Guide to selecting the right tennis racquet string tension for intermediate players.",
            gap_type="intent_reframe",
        ),
        Triplet(
            query="what happens if I eat too much sugar",
            positive_doc="Excessive dietary sucrose intake elevates hepatic de novo lipogenesis, promotes insulin resistance via chronic hyperinsulinemia, accelerates advanced glycation end-product formation, and increases cardiovascular disease risk biomarkers.",
            negative_doc="Step-by-step tutorial for creating animated SVG illustrations using Adobe Illustrator and CSS keyframes.",
            gap_type="intent_reframe",
        ),
        Triplet(
            query="can I run a marathon without training",
            positive_doc="Attempting 42.195 km without progressive endurance adaptation risks rhabdomyolysis, stress fractures, hyponatremia, and severe delayed-onset muscle soreness; a minimum 16-week periodised programme is medically recommended.",
            negative_doc="Comparison of noise-cancelling headphone models focused on Bluetooth codec support and battery life.",
            gap_type="intent_reframe",
        ),
        Triplet(
            query="is remote work here to stay",
            positive_doc="Post-pandemic labour market analysis indicates sustained hybrid work adoption: 58% of knowledge workers maintain flexible arrangements, driven by productivity metrics, reduced commercial real-estate footprints, and talent acquisition competitiveness.",
            negative_doc="Traditional Japanese tea ceremony procedures including utensil preparation and matcha whisking technique.",
            gap_type="intent_reframe",
        ),

        # ── 6. Additional mixed-gap examples ─────────────────────
        Triplet(
            query="good coffee near me",
            positive_doc="Specialty third-wave coffee roasters with single-origin pour-over bars, latte art competitions, and direct-trade sourcing transparency in urban metropolitan areas.",
            negative_doc="Industrial wastewater treatment processes including activated sludge and membrane bioreactor technologies.",
            gap_type="synonym",
        ),
        Triplet(
            query="how to not be nervous before a presentation",
            positive_doc="Public speaking anxiety management: diaphragmatic breathing exercises, cognitive restructuring of catastrophic thought patterns, systematic desensitisation through graduated exposure, and power-posing for pre-performance cortisol modulation.",
            negative_doc="Detailed specifications of the James Webb Space Telescope's near-infrared camera and mirror array.",
            gap_type="colloquial_to_professional",
        ),
        Triplet(
            query="enviornment friendly packaging",
            positive_doc="Sustainable packaging materials: compostable PLA bioplastics, mushroom mycelium moulded inserts, recycled corrugated fibreboard, and seaweed-based edible film wraps with lifecycle assessment data.",
            negative_doc="Troubleshooting common electrical wiring issues in residential circuit breaker panels.",
            gap_type="synonym",
        ),
        Triplet(
            query="让孩子爱上阅读的方法",
            positive_doc="Child literacy engagement strategies: creating a print-rich home environment, interactive read-aloud sessions with dialogic questioning, age-appropriate book selection using Lexile frameworks, and intrinsic motivation through reader identity reinforcement.",
            negative_doc="Configuring NGINX reverse proxy with SSL termination for containerised microservice architectures.",
            gap_type="cross_lingual",
        ),
        Triplet(
            query="how to deal with a toxic coworker",
            positive_doc="Workplace interpersonal conflict resolution: establishing professional boundaries through assertive communication, documentation of behavioural patterns for HR escalation, grey-rock emotional disengagement technique, and strategic alliance building.",
            negative_doc="Astrophysics primer on neutron star formation and pulsar emission mechanisms in binary star systems.",
            gap_type="colloquial_to_professional",
        ),
    ]
