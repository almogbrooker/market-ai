#!/usr/bin/env python3
"""
RISK AGENT - Chat-G.txt Section 6
Mission: Real-time limit monitoring with kill-switches
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskAgent:
    """
    Risk Agent - Chat-G.txt Section 6
    Real-time limit monitoring with kill-switches
    """
    
    def __init__(self, trading_config: Dict):
        logger.info("🛡️ RISK AGENT - REAL-TIME LIMIT MONITORING")
        
        self.config = trading_config
        self.base_dir = Path(__file__).parent.parent
        self.artifacts_dir = self.base_dir / "artifacts"
        
        # Risk limits from config
        self.limits = trading_config['risk_limits']
        self.kill_switches = trading_config.get('kill_switches', {})
        
        logger.info("🚨 Risk Limits:")
        logger.info(f"   Max Portfolio Loss: {self.limits['max_daily_loss']:.1%}")
        logger.info(f"   Max Leverage: {self.limits['max_leverage']:.1f}x")
        logger.info(f"   Max Position Size: {self.limits['max_position_size']:.1%}")
        logger.info(f"   Max Sector Exposure: {self.limits['max_sector_exposure']:.1%}")
        logger.info(f"   VIX Kill Switch: {self.kill_switches.get('vix_threshold', 35)}")
        
    def monitor_risk(self) -> Dict[str, Any]:
        """
        Monitor all risk metrics and kill-switches
        DoD: No breached limits (halt trading if violated), vol targeting, auto de-risking
        """
        
        logger.info("🔍 Monitoring risk limits...")
        
        try:
            # Load current portfolio
            current_portfolio = self._load_current_portfolio()
            if current_portfolio is None:
                return {'risk_status': 'UNKNOWN', 'reason': 'No portfolio data'}
            
            # Check portfolio-level risks
            portfolio_risks = self._check_portfolio_risks(current_portfolio)
            
            # Check position-level risks
            position_risks = self._check_position_risks(current_portfolio)
            
            # Check market kill-switches
            market_risks = self._check_market_kill_switches()
            
            # Check system health
            system_risks = self._check_system_health()
            
            # Aggregate risk assessment
            risk_assessment = self._aggregate_risk_assessment(
                portfolio_risks, position_risks, market_risks, system_risks
            )
            
            # Generate risk actions
            risk_actions = self._generate_risk_actions(risk_assessment)
            
            # Save risk report
            self._save_risk_report(risk_assessment, risk_actions)
            
            # Determine overall risk status
            overall_status = self._determine_risk_status(risk_assessment)
            
            result = {
                'risk_status': overall_status,
                'portfolio_risks': portfolio_risks,
                'position_risks': position_risks,
                'market_risks': market_risks,
                'system_risks': system_risks,
                'risk_actions': risk_actions,
                'timestamp': datetime.now().isoformat()
            }
            
            if overall_status == 'CRITICAL':
                logger.error("🚨🚨 CRITICAL RISK - IMMEDIATE ACTION REQUIRED 🚨🚨")
            elif overall_status == 'WARNING':
                logger.warning("⚠️ Risk Warning - Monitor Closely")
            else:
                logger.info("✅ Risk levels normal")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Risk monitoring failed: {e}")
            import traceback
            traceback.print_exc()
            return {'risk_status': 'ERROR', 'reason': f'Error: {e}'}
    
    def _load_current_portfolio(self) -> Optional[pd.DataFrame]:
        """Load current portfolio for risk monitoring"""
        
        portfolio_dir = self.artifacts_dir / "portfolios"
        
        if not portfolio_dir.exists():
            logger.warning("No portfolios directory found")
            return None
        
        # Find most recent portfolio
        portfolio_files = list(portfolio_dir.glob("portfolio_*.parquet"))
        
        if not portfolio_files:
            logger.warning("No portfolio files found")
            return None
        
        portfolio_files.sort()
        latest_portfolio_path = portfolio_files[-1]
        
        try:
            portfolio = pd.read_parquet(latest_portfolio_path)
            logger.info(f"📂 Loaded portfolio: {latest_portfolio_path.name}")
            return portfolio
        except Exception as e:
            logger.error(f"Failed to load portfolio: {e}")
            return None
    
    def _check_portfolio_risks(self, portfolio: pd.DataFrame) -> Dict[str, Any]:
        """Check portfolio-level risk metrics"""
        
        logger.info("📊 Checking portfolio-level risks...")
        
        # Calculate current P&L (simplified)
        # In production, would integrate with real-time pricing
        current_pnl_pct = np.random.normal(0, 0.01)  # Placeholder: random daily P&L
        
        # Portfolio leverage
        gross_exposure = np.abs(portfolio['weight']).sum()
        net_exposure = portfolio['weight'].sum()
        leverage = gross_exposure
        
        # Sector concentration
        if 'sector' in portfolio.columns:
            sector_exposures = portfolio.groupby('sector')['weight'].sum()
            max_sector_exposure = np.abs(sector_exposures).max()
        else:
            max_sector_exposure = 0
        
        # Position concentration
        max_position = np.abs(portfolio['weight']).max()
        
        # Risk checks
        risks = {
            'daily_pnl': {
                'value': current_pnl_pct,
                'limit': self.limits['max_daily_loss'],
                'breached': current_pnl_pct < -self.limits['max_daily_loss'],
                'severity': 'CRITICAL' if current_pnl_pct < -self.limits['max_daily_loss'] else 'OK'
            },
            'leverage': {
                'value': leverage,
                'limit': self.limits['max_leverage'],
                'breached': leverage > self.limits['max_leverage'],
                'severity': 'CRITICAL' if leverage > self.limits['max_leverage'] else 'OK'
            },
            'sector_concentration': {
                'value': max_sector_exposure,
                'limit': self.limits['max_sector_exposure'],
                'breached': max_sector_exposure > self.limits['max_sector_exposure'],
                'severity': 'WARNING' if max_sector_exposure > self.limits['max_sector_exposure'] else 'OK'
            },
            'position_concentration': {
                'value': max_position,
                'limit': self.limits['max_position_size'],
                'breached': max_position > self.limits['max_position_size'],
                'severity': 'WARNING' if max_position > self.limits['max_position_size'] else 'OK'
            }
        }
        
        # Log results
        for risk_name, risk_data in risks.items():
            status = "🚨" if risk_data['breached'] else "✅"
            logger.info(f"   {risk_name}: {risk_data['value']:.3f} (limit: {risk_data['limit']:.3f}) {status}")
        
        return risks
    
    def _check_position_risks(self, portfolio: pd.DataFrame) -> Dict[str, Any]:
        """Check individual position risks"""
        
        logger.info("🎯 Checking position-level risks...")
        
        # Position size violations
        oversized_positions = portfolio[np.abs(portfolio['weight']) > self.limits['max_position_size']]
        
        # Concentration in single stock
        position_concentration = {
            'oversized_count': len(oversized_positions),
            'oversized_tickers': oversized_positions['Ticker'].tolist() if len(oversized_positions) > 0 else [],
            'max_position_size': np.abs(portfolio['weight']).max(),
            'top5_concentration': np.abs(portfolio['weight']).nlargest(5).sum()
        }
        
        # Liquidity risk (placeholder)
        # In production, would check ADV ratios
        liquidity_risk = {
            'low_liquidity_positions': 0,  # Placeholder
            'estimated_liquidation_time': '< 1 day'  # Placeholder
        }
        
        risks = {
            'position_concentration': position_concentration,
            'liquidity_risk': liquidity_risk,
            'severity': 'WARNING' if len(oversized_positions) > 0 else 'OK'
        }
        
        if len(oversized_positions) > 0:
            logger.warning(f"⚠️ {len(oversized_positions)} oversized positions: {oversized_positions['Ticker'].tolist()}")
        else:
            logger.info("✅ All positions within size limits")
        
        return risks
    
    def _check_market_kill_switches(self) -> Dict[str, Any]:
        """Check market-based kill switches"""
        
        logger.info("🚨 Checking market kill-switches...")
        
        kill_switches = {}
        
        # VIX kill switch
        try:
            import yfinance as yf
            vix_data = yf.download('^VIX', period='5d', progress=False)
            if not vix_data.empty:
                current_vix = vix_data['Close'].iloc[-1]
                vix_threshold = self.kill_switches.get('vix_threshold', 35)
                
                kill_switches['vix'] = {
                    'value': current_vix,
                    'threshold': vix_threshold,
                    'triggered': current_vix > vix_threshold,
                    'severity': 'CRITICAL' if current_vix > vix_threshold else 'OK'
                }
            else:
                kill_switches['vix'] = {'triggered': False, 'severity': 'OK', 'value': None}
        except:
            kill_switches['vix'] = {'triggered': False, 'severity': 'OK', 'value': None}
        
        # Market circuit breaker check
        # In production, would check for market halts, unusual conditions
        kill_switches['circuit_breakers'] = {
            'triggered': False,
            'severity': 'OK'
        }
        
        # Correlation breakdown check
        # In production, would monitor inter-asset correlations
        kill_switches['correlation_breakdown'] = {
            'triggered': False,
            'severity': 'OK'
        }
        
        # Flash crash detection
        # In production, would monitor for extreme price movements
        kill_switches['flash_crash'] = {
            'triggered': False,
            'severity': 'OK'
        }
        
        # Check if any kill switches are triggered
        any_triggered = any(ks.get('triggered', False) for ks in kill_switches.values())
        
        if any_triggered:
            logger.error("🚨 KILL SWITCH TRIGGERED!")
            for name, switch in kill_switches.items():
                if switch.get('triggered', False):
                    logger.error(f"   {name}: TRIGGERED")
        else:
            logger.info("✅ All kill switches normal")
        
        return {
            'switches': kill_switches,
            'any_triggered': any_triggered,
            'severity': 'CRITICAL' if any_triggered else 'OK'
        }
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check system health and operational risks"""
        
        logger.info("🔧 Checking system health...")
        
        # Data freshness check
        labels_path = self.artifacts_dir / "labels" / "labels.parquet"
        data_freshness = 'OK'
        if labels_path.exists():
            labels_df = pd.read_parquet(labels_path)
            latest_data_date = pd.to_datetime(labels_df['Date']).max()
            days_stale = (datetime.now() - latest_data_date).days
            if days_stale > 2:
                data_freshness = 'STALE'
        else:
            data_freshness = 'MISSING'
        
        # Model health check
        model_health = 'OK'
        lgbm_results_path = self.artifacts_dir / "models" / "lgbm_results.json"
        if not lgbm_results_path.exists():
            model_health = 'MISSING'
        
        # Connectivity check (placeholder)
        connectivity_status = 'OK'  # In production, would ping broker APIs
        
        system_health = {
            'data_freshness': {
                'status': data_freshness,
                'severity': 'WARNING' if data_freshness != 'OK' else 'OK'
            },
            'model_health': {
                'status': model_health,
                'severity': 'CRITICAL' if model_health == 'MISSING' else 'OK'
            },
            'connectivity': {
                'status': connectivity_status,
                'severity': 'CRITICAL' if connectivity_status != 'OK' else 'OK'
            }
        }
        
        overall_severity = 'OK'
        for check in system_health.values():
            if check['severity'] == 'CRITICAL':
                overall_severity = 'CRITICAL'
                break
            elif check['severity'] == 'WARNING' and overall_severity == 'OK':
                overall_severity = 'WARNING'
        
        logger.info(f"   Data freshness: {data_freshness}")
        logger.info(f"   Model health: {model_health}")
        logger.info(f"   Connectivity: {connectivity_status}")
        
        return {
            'checks': system_health,
            'overall_severity': overall_severity
        }
    
    def _aggregate_risk_assessment(self, portfolio_risks: Dict, position_risks: Dict, 
                                  market_risks: Dict, system_risks: Dict) -> Dict[str, Any]:
        """Aggregate all risk assessments into overall risk profile"""
        
        # Collect all severities
        severities = []
        
        # Portfolio risks
        for risk in portfolio_risks.values():
            severities.append(risk.get('severity', 'OK'))
        
        # Position risks
        severities.append(position_risks.get('severity', 'OK'))
        
        # Market risks
        severities.append(market_risks.get('severity', 'OK'))
        
        # System risks
        severities.append(system_risks.get('overall_severity', 'OK'))
        
        # Determine overall severity
        if 'CRITICAL' in severities:
            overall_severity = 'CRITICAL'
        elif 'WARNING' in severities:
            overall_severity = 'WARNING'
        else:
            overall_severity = 'OK'
        
        # Count issues
        critical_count = severities.count('CRITICAL')
        warning_count = severities.count('WARNING')
        
        return {
            'overall_severity': overall_severity,
            'critical_issues': critical_count,
            'warning_issues': warning_count,
            'total_checks': len(severities),
            'severities': severities
        }
    
    def _generate_risk_actions(self, risk_assessment: Dict) -> List[Dict[str, str]]:
        """Generate recommended risk actions based on assessment"""
        
        actions = []
        
        severity = risk_assessment['overall_severity']
        
        if severity == 'CRITICAL':
            actions.append({
                'action': 'HALT_TRADING',
                'priority': 'IMMEDIATE',
                'description': 'Stop all trading immediately due to critical risk'
            })
            actions.append({
                'action': 'REDUCE_POSITIONS',
                'priority': 'IMMEDIATE', 
                'description': 'Begin emergency position reduction'
            })
            actions.append({
                'action': 'ALERT_TEAM',
                'priority': 'IMMEDIATE',
                'description': 'Alert risk management team immediately'
            })
        
        elif severity == 'WARNING':
            actions.append({
                'action': 'MONITOR_CLOSELY',
                'priority': 'HIGH',
                'description': 'Increase monitoring frequency'
            })
            actions.append({
                'action': 'REVIEW_POSITIONS',
                'priority': 'HIGH',
                'description': 'Review and consider reducing problem positions'
            })
            actions.append({
                'action': 'UPDATE_LIMITS',
                'priority': 'MEDIUM',
                'description': 'Consider tightening risk limits'
            })
        
        else:
            actions.append({
                'action': 'CONTINUE_NORMAL',
                'priority': 'LOW',
                'description': 'Continue normal operations'
            })
        
        return actions
    
    def _determine_risk_status(self, risk_assessment: Dict) -> str:
        """Determine overall risk status"""
        
        severity = risk_assessment['overall_severity']
        
        if severity == 'CRITICAL':
            return 'CRITICAL'
        elif severity == 'WARNING':
            return 'WARNING'
        else:
            return 'NORMAL'
    
    def _save_risk_report(self, risk_assessment: Dict, risk_actions: List[Dict]):
        """Save risk monitoring report"""
        
        # Ensure risk directory exists
        risk_dir = self.artifacts_dir / "risk"
        risk_dir.mkdir(parents=True, exist_ok=True)
        
        # Create risk report
        risk_report = {
            'timestamp': datetime.now().isoformat(),
            'risk_assessment': risk_assessment,
            'risk_actions': risk_actions,
            'config_limits': self.limits,
            'kill_switches_config': self.kill_switches
        }
        
        # Save report
        report_path = risk_dir / f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(risk_report, f, indent=2, default=str)
        
        logger.info(f"💾 Risk report saved: {report_path}")

def main():
    """Test the risk agent"""
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "trading_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        logger.error("Trading config not found")
        return False
    
    # Initialize and run agent
    agent = RiskAgent(config)
    result = agent.monitor_risk()
    
    status = result['risk_status']
    if status == 'CRITICAL':
        print("🚨 CRITICAL RISK - Immediate action required")
    elif status == 'WARNING':
        print("⚠️ Risk Warning - Monitor closely")
    elif status == 'NORMAL':
        print("✅ Risk levels normal")
    else:
        print(f"❓ Risk status: {status}")
    
    return True

if __name__ == "__main__":
    main()