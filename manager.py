#!/usr/bin/env python3
"""
ORCHESTRATOR AGENT - Chat-G.txt Implementation
Owns task graph, configs, and runbook for NASDAQ Long/Short Alpha Program
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrchestatorAgent:
    """
    Orchestrator Agent - glue for all agents
    Chat-G.txt Section 0: Owns task graph, configs, and runbook
    """
    
    def __init__(self):
        logger.info("🎯 ORCHESTRATOR AGENT - NASDAQ LONG/SHORT ALPHA PROGRAM")
        
        # Base paths
        self.base_dir = Path(__file__).parent
        self.config_dir = self.base_dir / "config"
        self.agents_dir = self.base_dir / "agents"
        self.artifacts_dir = self.base_dir / "artifacts"
        
        # Load configurations
        self.trading_config = self._load_config("trading_config.json")
        self.data_config = self._load_config("data_config.json")
        self.model_config = self._load_config("model_config.json")
        
        logger.info(f"📋 Global Objective: {self.trading_config['global_objective']['target_return']:.1%} net annual returns")
        logger.info(f"🎯 Target Metrics: Sharpe ≥{self.trading_config['global_objective']['min_sharpe']}, Max DD ≤{self.trading_config['global_objective']['max_drawdown']:.1%}")
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration file"""
        config_path = self.config_dir / config_file
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def run_pipeline(self, mode: str = "full", **kwargs) -> bool:
        """
        Run the complete pipeline
        Chat-G.txt DoD: One-command runs for: ingest → build → train → validate → backtest → paper_trade → report
        """
        
        logger.info("=" * 80)
        logger.info("🚀 NASDAQ LONG/SHORT ALPHA PIPELINE")
        logger.info("=" * 80)
        
        try:
            if mode in ["full", "ingest"]:
                success = self._run_universe_data_agent()
                if not success:
                    return False
                    
            if mode in ["full", "build"]:
                success = self._run_labeling_agent()
                if not success:
                    return False
                    
            if mode in ["full", "train"]:
                success = self._run_modeling_agents()
                if not success:
                    return False
                    
            if mode in ["full", "validate"]:
                success = self._run_validation_agent()
                if not success:
                    return False
                    
            if mode in ["full", "backtest"]:
                success = self._run_backtest()
                if not success:
                    return False
                    
            if mode in ["full", "paper_trade"]:
                success = self._run_paper_trade()
                if not success:
                    return False
                    
            if mode in ["full", "report"]:
                success = self._run_reporting_agent()
                if not success:
                    return False
            
            logger.info("✅ Pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False
    
    def _run_universe_data_agent(self) -> bool:
        """Run Universe & Data Agent (Chat-G.txt Section 1)"""
        logger.info("📊 Running Universe & Data Agent...")
        
        try:
            from agents.universe_data_agent import UniverseDataAgent
            agent = UniverseDataAgent(self.data_config)
            return agent.build_daily_features()
        except ImportError:
            logger.error("Universe & Data Agent not implemented yet")
            return False
    
    def _run_labeling_agent(self) -> bool:
        """Run Labeling & Target Agent (Chat-G.txt Section 2)"""
        logger.info("🎯 Running Labeling & Target Agent...")
        
        try:
            from agents.labeling_agent import LabelingAgent
            agent = LabelingAgent(self.data_config)
            return agent.create_targets()
        except ImportError:
            logger.error("Labeling Agent not implemented yet")
            return False
    
    def _run_modeling_agents(self) -> bool:
        """Run Modeling Agents (Chat-G.txt Section 3)"""
        logger.info("🤖 Running Modeling Agents...")
        
        try:
            # 3a) Baseline Ranker (LightGBM)
            from agents.baseline_ranker_agent import BaselineRankerAgent
            baseline_agent = BaselineRankerAgent(self.model_config)
            if not baseline_agent.train_model():
                return False
            
            # 3b) Sequence Alpha (LSTM or PatchTST)
            from agents.sequence_alpha_agent import SequenceAlphaAgent
            sequence_agent = SequenceAlphaAgent(self.model_config)
            if not sequence_agent.train_model():
                return False
            
            # 3c) Meta-Ensemble & Calibration
            from agents.meta_ensemble_agent import MetaEnsembleAgent
            meta_agent = MetaEnsembleAgent(self.model_config)
            return meta_agent.train_ensemble()
            
        except ImportError:
            logger.error("Modeling Agents not implemented yet")
            return False
    
    def _run_validation_agent(self) -> bool:
        """Run Validation Agent (Chat-G.txt Section 4)"""
        logger.info("🔍 Running Validation Agent...")
        
        try:
            from agents.validation_agent import ValidationAgent
            agent = ValidationAgent(self.trading_config)
            return agent.validate_models()
        except ImportError:
            logger.error("Validation Agent not implemented yet")
            return False
    
    def _run_backtest(self) -> bool:
        """Run Portfolio & Execution Agent backtest (Chat-G.txt Section 5)"""
        logger.info("📈 Running Backtest...")
        
        try:
            from agents.portfolio_execution_agent import PortfolioExecutionAgent
            agent = PortfolioExecutionAgent(self.trading_config)
            return agent.run_backtest()
        except ImportError:
            logger.error("Portfolio & Execution Agent not implemented yet")
            return False
    
    def _run_paper_trade(self) -> bool:
        """Start paper trading"""
        logger.info("📝 Starting Paper Trade...")
        
        try:
            from agents.paper_trading_agent import PaperTradingAgent
            agent = PaperTradingAgent(self.trading_config)
            agent.start_trading()
            return True
        except ImportError:
            logger.error("Paper trading agent not implemented yet")
            return False
    
    def _run_reporting_agent(self) -> bool:
        """Run Monitoring & Reporting Agent (Chat-G.txt Section 7)"""
        logger.info("📊 Running Reporting Agent...")
        
        try:
            from agents.monitoring_reporting_agent import MonitoringReportingAgent
            agent = MonitoringReportingAgent(self.trading_config)
            return agent.generate_reports()
        except ImportError:
            logger.error("Reporting Agent not implemented yet")
            return False

def main():
    """Main entry point with CLI interface"""
    
    parser = argparse.ArgumentParser(description="NASDAQ Long/Short Alpha Program Orchestrator")
    
    # Command options
    parser.add_argument("command", choices=["build", "train", "validate", "backtest", "paper_trade", "full"],
                       help="Command to run")
    
    # Build options
    parser.add_argument("--config", type=str, help="Config file path")
    
    # Training options  
    parser.add_argument("--cv", choices=["purged", "standard"], default="purged",
                       help="Cross-validation method")
    parser.add_argument("--walkforward", choices=["monthly", "quarterly"], default="monthly",
                       help="Walk-forward frequency")
    
    # Backtest options
    parser.add_argument("--cost_model", choices=["realistic_v2", "minimal"], default="realistic_v2",
                       help="Cost model to use")
    
    # Paper trading options
    parser.add_argument("--paper", action="store_true", help="Run in paper trading mode")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = OrchestatorAgent()
    
    # Run command
    if args.command == "build":
        success = orchestrator._run_universe_data_agent()
    elif args.command == "train":
        success = orchestrator._run_modeling_agents()
    elif args.command == "validate":
        success = orchestrator._run_validation_agent()
    elif args.command == "backtest":
        success = orchestrator._run_backtest()
    elif args.command == "paper_trade":
        success = orchestrator._run_paper_trade()
    elif args.command == "full":
        success = orchestrator.run_pipeline("full")
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())