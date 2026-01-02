"""
Main Entry Point for Vital Signs Monitoring System

Run with: python -m src.main
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils import Config, setup_logger
from src.utils.camera import Camera, test_camera


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Vital Signs Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["camera-test", "face-detect", "age", "heart-rate", "dashboard", "all"],
        default="camera-test",
        help="Operation mode (default: camera-test)"
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device ID (overrides config)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger("main")
    logger.info("=" * 50)
    logger.info("Vital Signs Monitoring System")
    logger.info("=" * 50)
    
    # Load configuration
    config = Config()
    logger.info(f"Project: {config.get('project.name')}")
    logger.info(f"Version: {config.get('project.version')}")
    
    # Run selected mode
    if args.mode == "camera-test":
        logger.info("Running camera test...")
        test_camera()
        
    elif args.mode == "face-detect":
        logger.info("Running face detection...")
        # TODO: Import and run face detection module
        logger.warning("Face detection module not yet implemented")
        
    elif args.mode == "age":
        logger.info("Running age estimation...")
        # TODO: Import and run age estimation module
        logger.warning("Age estimation module not yet implemented")
        
    elif args.mode == "heart-rate":
        logger.info("Running heart rate detection...")
        # TODO: Import and run rPPG heart rate module
        logger.warning("Heart rate module not yet implemented")
        
    elif args.mode == "dashboard":
        logger.info("Starting FastAPI dashboard...")
        # TODO: Import and run dashboard
        logger.warning("Dashboard not yet implemented")
        
    elif args.mode == "all":
        logger.info("Running full vital signs monitoring...")
        # TODO: Run all modules together
        logger.warning("Full system not yet implemented")
    
    logger.info("Shutting down...")


if __name__ == "__main__":
    main()
