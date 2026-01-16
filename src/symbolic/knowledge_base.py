"""Medical Knowledge Base using OWL/RDF ontologies.

Provides structured access to medical knowledge including:
- Drug information (classes, mechanisms, indications)
- Medical conditions (symptoms, treatments)
- Anatomical relationships
- Treatment protocols

Uses RDFLib and Owlready2 for ontology management.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class Drug:
    """Represents a drug in the knowledge base."""
    
    name: str
    generic_name: str = ""
    drug_class: str = ""
    mechanism: str = ""
    indications: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    pregnancy_category: str = ""
    controlled_substance: bool = False
    requires_monitoring: list[str] = field(default_factory=list)


@dataclass
class Condition:
    """Represents a medical condition."""
    
    name: str
    icd10_code: str = ""
    category: str = ""
    symptoms: list[str] = field(default_factory=list)
    treatments: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    complications: list[str] = field(default_factory=list)


@dataclass
class Interaction:
    """Drug interaction from knowledge base."""
    
    drug_a: str
    drug_b: str
    severity: str
    mechanism: str
    clinical_effect: str
    management: str
    evidence_level: str = "moderate"


class MedicalKnowledgeBase:
    """Knowledge base for medical information.
    
    Provides querying capabilities over:
    - Drug ontology
    - Condition ontology
    - Drug-condition relationships
    - Drug-drug interactions
    
    Example:
        ```python
        kb = MedicalKnowledgeBase()
        await kb.load()
        
        # Query drug information
        aspirin = kb.get_drug("aspirin")
        print(aspirin.mechanism)
        
        # Find drugs for condition
        drugs = kb.find_drugs_for_condition("hypertension")
        
        # Check interaction
        interactions = kb.get_interactions("warfarin")
        ```
    """
    
    def __init__(self, ontology_path: str | Path | None = None) -> None:
        """Initialize knowledge base.
        
        Args:
            ontology_path: Path to OWL ontology file
        """
        config = settings.symbolic.knowledge_base
        self.ontology_path = Path(ontology_path or config.path)
        self.cache_enabled = config.cache_enabled
        
        # In-memory caches
        self._drugs: dict[str, Drug] = {}
        self._conditions: dict[str, Condition] = {}
        self._interactions: dict[str, list[Interaction]] = {}
        self._drug_class_map: dict[str, list[str]] = {}
        
        self._loaded = False
    
    async def load(self) -> None:
        """Load knowledge base from ontology and data files."""
        if self._loaded:
            return
        
        # Load drug data
        await self._load_drugs()
        
        # Load condition data
        await self._load_conditions()
        
        # Load interactions
        await self._load_interactions()
        
        self._loaded = True
        logger.info(
            "knowledge_base_loaded",
            drugs=len(self._drugs),
            conditions=len(self._conditions),
            interactions=sum(len(v) for v in self._interactions.values()),
        )
    
    async def _load_drugs(self) -> None:
        """Load drug data from files."""
        # Built-in drug database (would load from OWL in production)
        drugs_data = [
            Drug(
                name="aspirin",
                generic_name="acetylsalicylic acid",
                drug_class="NSAID",
                mechanism="Inhibits cyclooxygenase (COX-1 and COX-2), reducing prostaglandin synthesis",
                indications=["pain", "fever", "inflammation", "cardiovascular protection"],
                contraindications=["active bleeding", "peptic ulcer", "aspirin allergy", "children with viral infection"],
                side_effects=["GI bleeding", "tinnitus", "allergic reaction"],
                pregnancy_category="D",
            ),
            Drug(
                name="warfarin",
                generic_name="warfarin sodium",
                drug_class="Anticoagulant",
                mechanism="Inhibits vitamin K epoxide reductase, reducing clotting factor synthesis",
                indications=["atrial fibrillation", "DVT", "PE", "mechanical heart valve"],
                contraindications=["active bleeding", "pregnancy", "severe liver disease"],
                side_effects=["bleeding", "bruising", "skin necrosis"],
                pregnancy_category="X",
                requires_monitoring=["INR", "signs of bleeding"],
            ),
            Drug(
                name="ibuprofen",
                generic_name="ibuprofen",
                drug_class="NSAID",
                mechanism="Non-selective COX inhibitor",
                indications=["pain", "fever", "inflammation", "arthritis"],
                contraindications=["peptic ulcer", "kidney disease", "heart failure", "NSAID allergy"],
                side_effects=["GI upset", "kidney injury", "cardiovascular events"],
                pregnancy_category="C",
            ),
            Drug(
                name="metformin",
                generic_name="metformin hydrochloride",
                drug_class="Biguanide",
                mechanism="Decreases hepatic glucose production, increases insulin sensitivity",
                indications=["type 2 diabetes", "PCOS", "prediabetes"],
                contraindications=["kidney disease", "liver disease", "metabolic acidosis"],
                side_effects=["GI upset", "lactic acidosis", "vitamin B12 deficiency"],
                pregnancy_category="B",
            ),
            Drug(
                name="lisinopril",
                generic_name="lisinopril",
                drug_class="ACE Inhibitor",
                mechanism="Inhibits angiotensin-converting enzyme, reducing angiotensin II",
                indications=["hypertension", "heart failure", "diabetic nephropathy"],
                contraindications=["pregnancy", "angioedema history", "bilateral renal artery stenosis"],
                side_effects=["cough", "hyperkalemia", "angioedema"],
                pregnancy_category="D",
            ),
            Drug(
                name="amoxicillin",
                generic_name="amoxicillin",
                drug_class="Penicillin antibiotic",
                mechanism="Inhibits bacterial cell wall synthesis",
                indications=["bacterial infections", "H. pylori", "endocarditis prophylaxis"],
                contraindications=["penicillin allergy"],
                side_effects=["diarrhea", "rash", "allergic reaction"],
                pregnancy_category="B",
            ),
            Drug(
                name="omeprazole",
                generic_name="omeprazole",
                drug_class="Proton Pump Inhibitor",
                mechanism="Irreversibly inhibits H+/K+ ATPase in gastric parietal cells",
                indications=["GERD", "peptic ulcer", "H. pylori eradication", "Zollinger-Ellison"],
                contraindications=["rilpivirine use"],
                side_effects=["headache", "diarrhea", "vitamin B12 deficiency", "hypomagnesemia"],
                pregnancy_category="C",
            ),
            Drug(
                name="atorvastatin",
                generic_name="atorvastatin calcium",
                drug_class="HMG-CoA reductase inhibitor (Statin)",
                mechanism="Inhibits HMG-CoA reductase, reducing cholesterol synthesis",
                indications=["hyperlipidemia", "cardiovascular protection", "diabetes"],
                contraindications=["active liver disease", "pregnancy", "breastfeeding"],
                side_effects=["myalgia", "liver enzyme elevation", "rhabdomyolysis"],
                pregnancy_category="X",
            ),
        ]
        
        for drug in drugs_data:
            self._drugs[drug.name.lower()] = drug
            
            # Build class map
            if drug.drug_class:
                class_lower = drug.drug_class.lower()
                if class_lower not in self._drug_class_map:
                    self._drug_class_map[class_lower] = []
                self._drug_class_map[class_lower].append(drug.name.lower())
    
    async def _load_conditions(self) -> None:
        """Load condition data from files."""
        conditions_data = [
            Condition(
                name="hypertension",
                icd10_code="I10",
                category="cardiovascular",
                symptoms=["headache", "dizziness", "nosebleed", "shortness of breath"],
                treatments=["ACE inhibitors", "ARBs", "calcium channel blockers", "diuretics"],
                risk_factors=["obesity", "high sodium diet", "family history", "age"],
                complications=["stroke", "heart attack", "kidney disease", "heart failure"],
            ),
            Condition(
                name="type 2 diabetes",
                icd10_code="E11",
                category="endocrine",
                symptoms=["polyuria", "polydipsia", "fatigue", "blurred vision"],
                treatments=["metformin", "sulfonylureas", "insulin", "GLP-1 agonists"],
                risk_factors=["obesity", "family history", "sedentary lifestyle", "age"],
                complications=["neuropathy", "nephropathy", "retinopathy", "cardiovascular disease"],
            ),
            Condition(
                name="atrial fibrillation",
                icd10_code="I48",
                category="cardiovascular",
                symptoms=["palpitations", "fatigue", "dizziness", "shortness of breath"],
                treatments=["anticoagulants", "rate control", "rhythm control", "ablation"],
                risk_factors=["hypertension", "heart disease", "thyroid disease", "alcohol"],
                complications=["stroke", "heart failure", "cognitive decline"],
            ),
            Condition(
                name="peptic ulcer",
                icd10_code="K27",
                category="gastrointestinal",
                symptoms=["abdominal pain", "bloating", "nausea", "hematemesis"],
                treatments=["PPIs", "H2 blockers", "H. pylori eradication", "antacids"],
                risk_factors=["H. pylori", "NSAID use", "smoking", "stress"],
                complications=["bleeding", "perforation", "obstruction"],
            ),
        ]
        
        for condition in conditions_data:
            self._conditions[condition.name.lower()] = condition
    
    async def _load_interactions(self) -> None:
        """Load drug interaction data."""
        # Try to load from file if exists
        interactions_path = Path(settings.symbolic.rules.drug_interactions)
        if interactions_path.exists():
            try:
                with open(interactions_path) as f:
                    data = json.load(f)
                    for item in data.get("interactions", []):
                        interaction = Interaction(**item)
                        self._add_interaction(interaction)
                return
            except Exception as e:
                logger.warning("failed_to_load_interactions_file", error=str(e))
        
        # Built-in interactions database
        interactions_data = [
            Interaction(
                drug_a="warfarin",
                drug_b="aspirin",
                severity="major",
                mechanism="Both affect hemostasis; additive bleeding risk",
                clinical_effect="Significantly increased risk of bleeding, including fatal hemorrhage",
                management="Avoid combination if possible. If necessary, use lowest aspirin dose and monitor closely",
                evidence_level="high",
            ),
            Interaction(
                drug_a="warfarin",
                drug_b="ibuprofen",
                severity="major",
                mechanism="NSAID inhibits platelet function and may increase warfarin levels",
                clinical_effect="Increased bleeding risk and potential for GI hemorrhage",
                management="Avoid NSAIDs with warfarin. Use acetaminophen for pain if needed",
                evidence_level="high",
            ),
            Interaction(
                drug_a="lisinopril",
                drug_b="potassium",
                severity="moderate",
                mechanism="ACE inhibitors reduce potassium excretion",
                clinical_effect="Risk of hyperkalemia",
                management="Monitor potassium levels regularly; avoid potassium supplements",
                evidence_level="high",
            ),
            Interaction(
                drug_a="metformin",
                drug_b="contrast dye",
                severity="major",
                mechanism="Contrast may cause acute kidney injury, leading to metformin accumulation",
                clinical_effect="Risk of lactic acidosis",
                management="Hold metformin 48 hours before and after contrast administration",
                evidence_level="high",
            ),
            Interaction(
                drug_a="atorvastatin",
                drug_b="grapefruit",
                severity="moderate",
                mechanism="Grapefruit inhibits CYP3A4 metabolism of atorvastatin",
                clinical_effect="Increased statin levels and risk of myopathy",
                management="Limit grapefruit intake; monitor for muscle pain",
                evidence_level="moderate",
            ),
            Interaction(
                drug_a="omeprazole",
                drug_b="clopidogrel",
                severity="major",
                mechanism="Omeprazole inhibits CYP2C19, reducing clopidogrel activation",
                clinical_effect="Reduced antiplatelet effect; increased cardiovascular events",
                management="Use alternative PPI (pantoprazole) or H2 blocker",
                evidence_level="high",
            ),
        ]
        
        for interaction in interactions_data:
            self._add_interaction(interaction)
    
    def _add_interaction(self, interaction: Interaction) -> None:
        """Add an interaction to the database."""
        drug_a = interaction.drug_a.lower()
        drug_b = interaction.drug_b.lower()
        
        if drug_a not in self._interactions:
            self._interactions[drug_a] = []
        if drug_b not in self._interactions:
            self._interactions[drug_b] = []
        
        self._interactions[drug_a].append(interaction)
        self._interactions[drug_b].append(interaction)
    
    def get_drug(self, name: str) -> Drug | None:
        """Get drug by name.
        
        Args:
            name: Drug name (case-insensitive)
        
        Returns:
            Drug object or None if not found
        """
        return self._drugs.get(name.lower())
    
    def get_condition(self, name: str) -> Condition | None:
        """Get condition by name.
        
        Args:
            name: Condition name (case-insensitive)
        
        Returns:
            Condition object or None if not found
        """
        return self._conditions.get(name.lower())
    
    def get_interactions(self, drug: str) -> list[Interaction]:
        """Get all interactions for a drug.
        
        Args:
            drug: Drug name
        
        Returns:
            List of interactions
        """
        return self._interactions.get(drug.lower(), [])
    
    def get_drugs_by_class(self, drug_class: str) -> list[Drug]:
        """Get all drugs in a class.
        
        Args:
            drug_class: Drug class name
        
        Returns:
            List of drugs in that class
        """
        drug_names = self._drug_class_map.get(drug_class.lower(), [])
        return [self._drugs[name] for name in drug_names if name in self._drugs]
    
    def find_drugs_for_condition(self, condition: str) -> list[Drug]:
        """Find drugs indicated for a condition.
        
        Args:
            condition: Condition name
        
        Returns:
            List of drugs indicated for this condition
        """
        condition_lower = condition.lower()
        matching_drugs = []
        
        for drug in self._drugs.values():
            for indication in drug.indications:
                if condition_lower in indication.lower():
                    matching_drugs.append(drug)
                    break
        
        return matching_drugs
    
    def check_contraindication(self, drug: str, condition: str) -> bool:
        """Check if drug is contraindicated for a condition.
        
        Args:
            drug: Drug name
            condition: Condition name
        
        Returns:
            True if contraindicated
        """
        drug_obj = self.get_drug(drug)
        if not drug_obj:
            return False
        
        condition_lower = condition.lower()
        for contraindication in drug_obj.contraindications:
            if condition_lower in contraindication.lower():
                return True
        
        return False
    
    def query(self, query_type: str, **kwargs: Any) -> list[Any]:
        """Execute a query against the knowledge base.
        
        Args:
            query_type: Type of query (drug, condition, interaction, etc.)
            **kwargs: Query parameters
        
        Returns:
            List of matching entities
        """
        if query_type == "drug":
            name = kwargs.get("name")
            if name:
                drug = self.get_drug(name)
                return [drug] if drug else []
            drug_class = kwargs.get("drug_class")
            if drug_class:
                return self.get_drugs_by_class(drug_class)
            return list(self._drugs.values())
        
        elif query_type == "condition":
            name = kwargs.get("name")
            if name:
                condition = self.get_condition(name)
                return [condition] if condition else []
            return list(self._conditions.values())
        
        elif query_type == "interaction":
            drug = kwargs.get("drug")
            if drug:
                return self.get_interactions(drug)
            return []
        
        else:
            logger.warning("unknown_query_type", query_type=query_type)
            return []
    
    def to_dict(self) -> dict[str, Any]:
        """Export knowledge base to dictionary."""
        return {
            "drugs": {k: v.__dict__ for k, v in self._drugs.items()},
            "conditions": {k: v.__dict__ for k, v in self._conditions.items()},
            "interactions": {k: [i.__dict__ for i in v] for k, v in self._interactions.items()},
        }
