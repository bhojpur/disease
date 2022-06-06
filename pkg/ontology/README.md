# Bhojpur Disease - Ontology Database

We have no intention of duplicating this work.

## Single Parentage Ontology

The disease classification is represented as both a single `is_a` hierarchy (`doid-non-classified.obo`).

## Multi-Parent Disease Classification

A *logically* inferred multiple parentage classification is produced as `doid.obo` file. It integrates
logical definitions defining the anatomical location or `cell` or `origin` of a disease.

The `HumanDO.obo` file is equivalent to the `doid-non-classified.obo` files produced in this repository.

## Ontology Files

### Logical Definition

- **Anatomy**: UBERON; `Cell Type of Origin`: Cell Ontology (CL); `Taxonomy`: NCBITaxon
- **Symptoms**: Symptom `Ontology`: SYMP; `Phenotype`: Human Phenotype Ontology (HPO)
- **Transmission Process**: Pathogen Transmission Methods (TRANS)
- **Sequence features**: Sequence Ontology (SO); `RO` (Relations Ontology)
- **FoodON**: allergic triggers; `CHEBI`: chemical triggers/environmental triggers/drivers
- **BRENDA tissue/ enzyme source (BTO)**: allergic triggers

## References

The OBO Foundry products

- [doid.owl](https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/master/src/ontology/doid.owl)
- [doid.obo](https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/master/src/ontology/doid.obo)
