"""
main.py
Full pipeline entrypoint: train → evaluate → (optionally) monitor.

Usage:
    python madewithml/main.py \
        --experiment-name "baseline" \
        --dataset-loc data/dataset.csv
"""

from madewithml.config import logger
from madewithml.train import train_model
from madewithml.evaluate import evaluate_model


def run_pipeline(
    experiment_name: str = "mlops_experiment",
    dataset_loc: str = "data/dataset.csv",
    num_epochs: int = 10,
):
    logger.info("═══════════════════════════════════════")
    logger.info(" MLOps Pipeline Starting")
    logger.info("═══════════════════════════════════════")

    logger.info("Step 1/2 — Training ...")
    train_model(
        experiment_name=experiment_name,
        dataset_loc=dataset_loc,
        num_epochs=num_epochs,
    )

    logger.info("Step 2/2 — Evaluation ...")
    evaluate_model(
        experiment_name=experiment_name,
        dataset_loc=dataset_loc,
    )

    logger.info("═══════════════════════════════════════")
    logger.info(" Pipeline Complete ✅")
    logger.info("═══════════════════════════════════════")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="mlops_experiment")
    parser.add_argument("--dataset-loc", default="data/dataset.csv")
    parser.add_argument("--num-epochs", type=int, default=10)
    args = parser.parse_args()
    run_pipeline(
        experiment_name=args.experiment_name,
        dataset_loc=args.dataset_loc,
        num_epochs=args.num_epochs,
    )
