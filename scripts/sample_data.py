#!/usr/bin/env python3
"""
Generate sample documents for testing the temporal knowledge graph system.

This script creates several sample text documents with different topics
to demonstrate the capabilities of the temporal knowledge graph.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_kg_rag.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


SAMPLE_DOCUMENTS = {
    "artificial_intelligence_2023.txt": """
# Artificial Intelligence in 2023

Artificial Intelligence (AI) has made remarkable progress in 2023. Major tech companies like Google,
OpenAI, and Meta have released groundbreaking models.

## Large Language Models

OpenAI's GPT-4 was released in March 2023, demonstrating unprecedented capabilities in natural language
understanding and generation. The model shows significant improvements in reasoning, coding, and
multi-modal understanding compared to its predecessors.

Google responded with Bard, their conversational AI service, and later announced Gemini as their
most capable model. Microsoft integrated GPT-4 into their Bing search engine and Office products.

## Computer Vision

Computer vision technology has advanced significantly, with applications in healthcare, autonomous
vehicles, and security systems. Companies like Tesla and Waymo continue to push the boundaries of
self-driving technology.

## Ethical Considerations

As AI systems become more powerful, concerns about AI safety, bias, and job displacement have
intensified. Researchers and policymakers are working on frameworks for responsible AI development.

The European Union proposed the AI Act, aiming to regulate AI systems based on their risk level.
This legislation could set a global precedent for AI governance.
    """,

    "climate_change_research.txt": """
# Climate Change Research: Recent Findings

Climate change remains one of the most pressing challenges facing humanity. Recent research from
NASA and the IPCC has provided new insights into the pace and impact of global warming.

## Temperature Trends

Global average temperatures have risen by approximately 1.1°C since pre-industrial times.
Scientists from the University of Oxford and MIT warn that we are approaching critical tipping points.

## Arctic Ice Melting

The Arctic region is warming at twice the global average rate. Research led by Dr. Jennifer Francis
at the Woods Hole Research Center shows dramatic reductions in sea ice extent.

## Impact on Ecosystems

Climate change is affecting ecosystems worldwide. The Great Barrier Reef in Australia has experienced
severe bleaching events. Amazon rainforest degradation, studied by researchers in Brazil, poses risks
to global carbon storage.

## Renewable Energy Solutions

Countries like Germany, Denmark, and China are leading the transition to renewable energy.
Solar and wind power costs have declined dramatically, making them competitive with fossil fuels.

## International Cooperation

The Paris Agreement, signed by 196 countries, aims to limit global warming to well below 2°C.
However, current commitments fall short of what's needed to meet this goal.
    """,

    "quantum_computing_advances.txt": """
# Quantum Computing Advances

Quantum computing represents a paradigm shift in computation, with potential applications in
cryptography, drug discovery, and optimization problems.

## Major Players

IBM, Google, and Microsoft are investing heavily in quantum computing research. In 2023,
IBM announced their Condor processor with over 1,000 qubits, while Google continues work on
quantum error correction.

## Quantum Supremacy

Google claimed quantum supremacy in 2019, demonstrating that their Sycamore processor could
solve a specific problem faster than classical supercomputers. This achievement, led by
Dr. John Martinis, marked a milestone in quantum computing.

## Applications in Drug Discovery

Pharmaceutical companies like Pfizer and Roche are exploring quantum computing for molecular
simulation. This could revolutionize drug discovery by enabling accurate simulation of
molecular interactions.

## Challenges

Quantum systems are extremely fragile and require near-absolute zero temperatures. Researchers
at institutions like MIT, Caltech, and ETH Zurich are working on error correction and
increasing coherence times.

## Future Outlook

While practical quantum advantage for real-world problems may still be years away, the field
is progressing rapidly. Companies like IonQ and Rigetti are developing quantum cloud services
to make the technology more accessible.
    """,

    "machine_learning_healthcare.txt": """
# Machine Learning Applications in Healthcare

Machine learning is transforming healthcare delivery, diagnosis, and treatment planning.
Institutions like Mayo Clinic and Stanford Medicine are at the forefront of this revolution.

## Medical Imaging

AI systems developed by companies like Zebra Medical Vision and Aidoc can detect abnormalities
in medical images with accuracy rivaling or exceeding human radiologists. These systems can
identify tumors, fractures, and other conditions in X-rays, CT scans, and MRIs.

## Drug Discovery

DeepMind's AlphaFold has revolutionized protein structure prediction, potentially accelerating
drug discovery. Researchers at Harvard Medical School and Johns Hopkins are using machine
learning to identify potential drug candidates and predict their effectiveness.

## Personalized Medicine

Machine learning enables analysis of genomic data to tailor treatments to individual patients.
Companies like 23andMe and Tempus are building platforms for personalized healthcare.

## Disease Prediction

Predictive models can identify patients at risk of developing conditions like diabetes,
heart disease, or sepsis. Researchers at Mount Sinai Hospital have developed systems that
can predict patient deterioration hours before it occurs.

## Challenges and Ethics

Privacy concerns, algorithmic bias, and the need for interpretable AI remain significant
challenges. The FDA is developing frameworks for regulating AI-based medical devices.
Researchers at institutions like Princeton and Berkeley are working on fairness and
transparency in healthcare AI.
    """,
}


def create_sample_documents(output_dir: Path) -> None:
    """
    Create sample documents in the specified directory.

    Args:
        output_dir: Directory to create documents in
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating {len(SAMPLE_DOCUMENTS)} sample documents in {output_dir}")

    for filename, content in SAMPLE_DOCUMENTS.items():
        file_path = output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip())

        logger.info(f"  Created: {filename} ({len(content)} characters)")

    logger.info(f"\n✓ Sample documents created successfully!")
    logger.info(f"\nTo ingest these documents, run:")
    logger.info(f"  python scripts/ingest_documents.py --path {output_dir} --pattern '*.txt'")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate sample documents for testing"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./sample_data",
        help="Output directory for sample documents (default: ./sample_data)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    logger.info("=" * 60)
    logger.info("Sample Data Generator")
    logger.info("=" * 60)

    try:
        output_dir = Path(args.output_dir).resolve()
        create_sample_documents(output_dir)
        return 0

    except Exception as e:
        logger.error(f"Failed to create sample documents: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
