"""Model lifecycle management script for promoting models between environments."""

import click
import logging
import json
from pathlib import Path
from config import MLOpsConfig
from mlflow_manager import MLflowManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Model lifecycle management CLI."""
    pass


@cli.command()
@click.option('--mlflow-uri', type=str, default='./mlruns', help='MLflow tracking URI')
@click.option('--model-name', type=str, default='iris-classifier', help='Model name')
@click.option('--from-env', type=click.Choice(['development', 'staging', 'production']),
              required=True, help='Source environment')
@click.option('--to-env', type=click.Choice(['development', 'staging', 'production']),
              required=True, help='Target environment')
@click.option('--version', type=str, help='Specific model version to promote (latest if not specified)')
def promote(mlflow_uri: str, model_name: str, from_env: str, to_env: str, version: str):
    """Promote model from one environment to another."""
    
    config = MLOpsConfig(mlflow_tracking_uri=mlflow_uri)
    manager = MLflowManager(config)
    
    logger.info(f"Promoting model '{model_name}' from {from_env} to {to_env}")
    
    try:
        new_version = manager.promote_model(model_name, from_env, to_env, version)
        logger.info(f"✅ Model promoted successfully!")
        logger.info(f"   Source: {model_name}_{from_env} (v{version or 'latest'})")
        logger.info(f"   Target: {model_name}_{to_env} (v{new_version})")
        
    except Exception as e:
        logger.error(f"❌ Promotion failed: {e}")
        raise


@cli.command()
@click.option('--mlflow-uri', type=str, default='./mlruns', help='MLflow tracking URI')
@click.option('--model-name', type=str, default='iris-classifier', help='Model name')
@click.option('--environment', type=click.Choice(['development', 'staging', 'production']),
              default='production', help='Environment')
@click.option('--version', type=str, required=True, help='Model version')
@click.option('--stage', type=click.Choice(['Staging', 'Production', 'Archived']),
              required=True, help='Target stage')
def transition(mlflow_uri: str, model_name: str, environment: str, version: str, stage: str):
    """Transition model to different stage."""
    
    config = MLOpsConfig(mlflow_tracking_uri=mlflow_uri)
    manager = MLflowManager(config)
    
    logger.info(f"Transitioning {model_name}_{environment} v{version} to {stage}")
    
    try:
        manager.transition_model_stage(model_name, version, stage, environment)
        logger.info(f"✅ Model transitioned to {stage} successfully!")
        
    except Exception as e:
        logger.error(f"❌ Transition failed: {e}")
        raise


@cli.command()
@click.option('--mlflow-uri', type=str, default='./mlruns', help='MLflow tracking URI')
@click.option('--model-name', type=str, default='iris-classifier', help='Model name')
@click.option('--output-file', type=str, help='Output file for the report (JSON)')
def compare(mlflow_uri: str, model_name: str, output_file: str):
    """Compare model performance across environments."""
    
    config = MLOpsConfig(mlflow_tracking_uri=mlflow_uri)
    manager = MLflowManager(config)
    
    logger.info(f"Comparing model performance for '{model_name}' across environments")
    
    try:
        comparison = manager.get_model_performance_comparison(model_name)
        
        # Print comparison table
        print("\n🔍 Model Performance Comparison")
        print("=" * 80)
        
        for env, data in comparison.items():
            print(f"\n📊 {env.upper()} Environment:")
            if 'error' in data:
                print(f"   ❌ Error: {data['error']}")
            else:
                print(f"   Version: {data.get('version', 'N/A')}")
                print(f"   Stage: {data.get('stage', 'N/A')}")
                print(f"   Test Accuracy: {data.get('metrics', {}).get('test_accuracy', 'N/A'):.4f}")
                print(f"   Test F1 Score: {data.get('metrics', {}).get('test_f1_score', 'N/A'):.4f}")
                print(f"   CV Mean Accuracy: {data.get('metrics', {}).get('cv_mean_accuracy', 'N/A'):.4f}")
        
        print("\n" + "=" * 80)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            logger.info(f"📄 Detailed comparison saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        raise


@cli.command()
@click.option('--mlflow-uri', type=str, default='./mlruns', help='MLflow tracking URI')
@click.option('--model-name', type=str, default='iris-classifier', help='Model name')
@click.option('--output-file', type=str, help='Output file for the report (JSON)')
def report(mlflow_uri: str, model_name: str, output_file: str):
    """Generate comprehensive model report."""
    
    config = MLOpsConfig(mlflow_tracking_uri=mlflow_uri)
    manager = MLflowManager(config)
    
    logger.info(f"Generating comprehensive report for '{model_name}'")
    
    try:
        report_data = manager.generate_model_report(model_name)
        
        # Print summary
        print(f"\n📋 Model Report: {model_name}")
        print("=" * 80)
        print(f"Generated: {report_data['generated_at']}")
        
        for env, data in report_data['environments'].items():
            print(f"\n🏗️  {env.upper()} Environment:")
            if 'error' in data:
                print(f"   ❌ Error: {data['error']}")
            else:
                print(f"   Total Versions: {data.get('total_versions', 0)}")
                
                # Show latest version details
                versions = data.get('versions', [])
                if versions:
                    latest = max(versions, key=lambda x: x.get('version', '0'))
                    print(f"   Latest Version: {latest.get('version')}")
                    print(f"   Stage: {latest.get('stage')}")
                    metrics = latest.get('metrics', {})
                    if metrics:
                        print(f"   Test Accuracy: {metrics.get('test_accuracy', 'N/A')}")
                        print(f"   Test F1: {metrics.get('test_f1_score', 'N/A')}")
        
        print("\n" + "=" * 80)
        
        # Save detailed report
        if output_file:
            output_path = output_file
        else:
            output_path = f"model_report_{model_name}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"📄 Detailed report saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        raise


@cli.command()
@click.option('--mlflow-uri', type=str, default='./mlruns', help='MLflow tracking URI')
@click.option('--model-name', type=str, default='iris-classifier', help='Model name')
def status(mlflow_uri: str, model_name: str):
    """Show current model status across all environments."""
    
    config = MLOpsConfig(mlflow_tracking_uri=mlflow_uri)
    manager = MLflowManager(config)
    
    logger.info(f"Checking status for model '{model_name}'")
    
    try:
        # Get production model status
        prod_model = manager.get_production_model(model_name)
        
        print(f"\n🚀 Production Model Status")
        print("=" * 50)
        
        if prod_model:
            print(f"✅ Production model is available")
            print(f"   Version: {prod_model['version']}")
            print(f"   Stage: {prod_model['stage']}")
            print(f"   Model URI: {prod_model['model_uri']}")
        else:
            print("❌ No production model found")
        
        # Get performance comparison
        comparison = manager.get_model_performance_comparison(model_name)
        
        print(f"\n📊 Environment Overview")
        print("=" * 50)
        
        for env in ['development', 'staging', 'production']:
            data = comparison.get(env, {})
            status_icon = "✅" if 'version' in data else "❌"
            
            print(f"{status_icon} {env.upper()}: ", end="")
            
            if 'error' in data:
                print(f"Error - {data['error']}")
            elif 'version' in data:
                acc = data.get('metrics', {}).get('test_accuracy', 0)
                print(f"v{data['version']} ({data['stage']}) - Accuracy: {acc:.3f}")
            else:
                print("No model found")
        
        print()
        
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        raise


@cli.command()
@click.option('--mlflow-uri', type=str, default='./mlruns', help='MLflow tracking URI')
@click.option('--environment', type=click.Choice(['development', 'staging', 'production']),
              required=True, help='Environment to clean')
@click.option('--keep-last', type=int, default=10, help='Number of runs to keep')
@click.option('--confirm', is_flag=True, help='Confirm deletion without prompt')
def cleanup(mlflow_uri: str, environment: str, keep_last: int, confirm: bool):
    """Clean up old experiment runs to save space."""
    
    config = MLOpsConfig(mlflow_tracking_uri=mlflow_uri)
    manager = MLflowManager(config)
    
    if not confirm:
        response = click.confirm(
            f"Are you sure you want to delete old runs in {environment} environment? "
            f"(keeping last {keep_last} runs)"
        )
        if not response:
            logger.info("Cleanup cancelled")
            return
    
    logger.info(f"Cleaning up old runs in {environment} environment (keeping last {keep_last})")
    
    try:
        manager.cleanup_old_runs(environment, keep_last)
        logger.info(f"✅ Cleanup completed for {environment} environment")
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        raise


if __name__ == "__main__":
    cli()