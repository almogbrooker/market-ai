#!/usr/bin/env python3
"""
DECISION LOGGING SYSTEM
Detailed logging for every trading decision with full context
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

class DecisionLogger:
    """Comprehensive decision logging system"""
    
    def __init__(self, log_dir="PRODUCTION/logs/decisions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session log file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = self.log_dir / f"session_{self.session_id}.json"
        self.decisions_log = self.log_dir / f"decisions_{self.session_id}.csv"
        
        # Initialize session
        self.session_data = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "decisions": [],
            "liquidations": [],
            "orders": [],
            "errors": []
        }
        
        self.decision_records = []
        
    def log_liquidation_decision(self, position, reason="daily_reset"):
        """Log position liquidation decision"""
        liquidation = {
            "timestamp": datetime.now().isoformat(),
            "action": "LIQUIDATE",
            "symbol": position.get('symbol', 'UNKNOWN'),
            "current_qty": position.get('qty', 0),
            "current_side": position.get('side', 'unknown'),
            "market_value": position.get('market_value', 0),
            "unrealized_pl": position.get('unrealized_pl', 0),
            "reason": reason,
            "decision_context": {
                "strategy": "Daily portfolio reset",
                "rule": "Liquidate all positions before new signals",
                "confidence": 1.0
            }
        }
        
        self.session_data["liquidations"].append(liquidation)
        print(f"📊 LIQUIDATION: {liquidation['symbol']} {liquidation['current_side']} {liquidation['current_qty']} shares")
        print(f"   Reason: {reason}")
        print(f"   P&L: ${float(liquidation['unrealized_pl']):.2f}")
        
    def log_signal_generation(self, symbol, raw_prediction, gate_passed, features_used):
        """Log signal generation decision"""
        decision = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "raw_prediction": float(raw_prediction),
            "gate_passed": bool(gate_passed),
            "signal_strength": abs(float(raw_prediction)),
            "features_used": features_used,
            "decision_stage": "signal_generation"
        }
        
        self.decision_records.append(decision)
        
        status = "✅ PASS" if gate_passed else "❌ REJECT"
        print(f"🎯 SIGNAL {symbol}: {status}")
        print(f"   Prediction: {raw_prediction:.6f}")
        print(f"   Signal Strength: {abs(raw_prediction):.6f}")
        print(f"   Gate Status: {'Passed' if gate_passed else 'Rejected'}")
        
    def log_position_sizing_decision(self, symbol, prediction, portfolio_value, position_size_pct, dollar_amount, shares):
        """Log position sizing decision"""
        decision = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "prediction": float(prediction),
            "portfolio_value": float(portfolio_value),
            "position_size_pct": float(position_size_pct),
            "dollar_amount": float(dollar_amount),
            "calculated_shares": int(shares),
            "decision_stage": "position_sizing",
            "sizing_logic": {
                "base_size": "3% of portfolio per position",
                "risk_adjustment": "None applied",
                "confidence_scaling": "None applied"
            }
        }
        
        self.decision_records.append(decision)
        print(f"📏 SIZING {symbol}:")
        print(f"   Portfolio Value: ${portfolio_value:,.2f}")
        print(f"   Position Size: {position_size_pct:.1%} = ${dollar_amount:,.2f}")
        print(f"   Shares: {shares}")
        
    def log_order_decision(self, symbol, side, qty, current_price, order_type="market"):
        """Log order execution decision"""
        order = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "quantity": int(qty),
            "price": float(current_price),
            "order_type": order_type,
            "estimated_value": float(current_price) * int(qty),
            "decision_context": {
                "execution_strategy": "Market order for immediate fill",
                "timing": "During trading hours",
                "liquidity_check": "Assumed sufficient for large-cap stocks"
            }
        }
        
        self.session_data["orders"].append(order)
        print(f"📋 ORDER {symbol}: {side} {qty} shares @ ${current_price:.2f}")
        print(f"   Estimated Value: ${order['estimated_value']:,.2f}")
        print(f"   Order Type: {order_type}")
        
    def log_risk_management_decision(self, total_positions, total_exposure, risk_limits):
        """Log risk management decisions"""
        risk_decision = {
            "timestamp": datetime.now().isoformat(),
            "total_positions": total_positions,
            "total_exposure_pct": float(total_exposure),
            "risk_limits": risk_limits,
            "decision_stage": "risk_management",
            "risk_checks": {
                "exposure_check": total_exposure <= risk_limits.get("baseline_gross_exposure", 0.33),
                "position_count_check": total_positions <= 20,
                "individual_position_check": "3% max per position enforced"
            }
        }
        
        self.decision_records.append(risk_decision)
        print(f"🛡️ RISK MANAGEMENT:")
        print(f"   Total Positions: {total_positions}")
        print(f"   Total Exposure: {total_exposure:.1%}")
        print(f"   Baseline Limit: {risk_limits.get('baseline_gross_exposure', 0.33):.1%}")
        print(f"   Status: {'✅ WITHIN LIMITS' if total_exposure <= risk_limits.get('baseline_gross_exposure', 0.33) else '⚠️ OVER LIMITS'}")
        
    def log_gate_configuration(self, gate_config):
        """Log conformal gate configuration"""
        gate_info = {
            "timestamp": datetime.now().isoformat(),
            "gate_method": gate_config.get("method", "unknown"),
            "threshold": gate_config.get("abs_score_threshold", 0),
            "target_accept_rate": gate_config.get("target_accept_rate", 0),
            "actual_accept_rate": gate_config.get("actual_accept_rate", 0),
            "decision_stage": "gate_configuration"
        }
        
        self.session_data["gate_config"] = gate_info
        print(f"🚪 GATE CONFIG:")
        print(f"   Method: {gate_info['gate_method']}")
        print(f"   Threshold: {gate_info['threshold']:.6f}")
        print(f"   Target Accept Rate: {gate_info['target_accept_rate']:.1%}")
        print(f"   Actual Accept Rate: {gate_info['actual_accept_rate']:.1%}")
        
    def log_model_info(self, model_path, features_count, model_config):
        """Log model information"""
        model_info = {
            "timestamp": datetime.now().isoformat(),
            "model_path": str(model_path),
            "features_count": features_count,
            "model_config": model_config,
            "decision_stage": "model_loading"
        }
        
        self.session_data["model_info"] = model_info
        print(f"🧠 MODEL INFO:")
        print(f"   Model Path: {model_path}")
        print(f"   Features: {features_count}")
        print(f"   Architecture: {model_config.get('size_config', {})}")
        
    def log_error(self, error_type, error_message, context=None):
        """Log errors with context"""
        error = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": str(error_message),
            "context": context or {}
        }
        
        self.session_data["errors"].append(error)
        print(f"❌ ERROR [{error_type}]: {error_message}")
        if context:
            print(f"   Context: {context}")
            
    def save_session_summary(self, final_portfolio_value=None, total_orders=None):
        """Save complete session summary"""
        self.session_data["end_time"] = datetime.now().isoformat()
        
        if final_portfolio_value:
            self.session_data["final_portfolio_value"] = float(final_portfolio_value)
        if total_orders:
            self.session_data["total_orders_executed"] = int(total_orders)
            
        # Summary statistics
        self.session_data["summary"] = {
            "liquidations_count": len(self.session_data["liquidations"]),
            "orders_count": len(self.session_data["orders"]),
            "errors_count": len(self.session_data["errors"]),
            "decisions_count": len(self.decision_records)
        }
        
        # Save session JSON
        with open(self.session_log, 'w') as f:
            json.dump(self.session_data, f, indent=2)
            
        # Save decisions CSV
        if self.decision_records:
            df = pd.DataFrame(self.decision_records)
            df.to_csv(self.decisions_log, index=False)
            
        print(f"\n📊 SESSION SUMMARY:")
        print(f"   Session ID: {self.session_id}")
        print(f"   Liquidations: {self.session_data['summary']['liquidations_count']}")
        print(f"   Orders: {self.session_data['summary']['orders_count']}")
        print(f"   Decisions Logged: {self.session_data['summary']['decisions_count']}")
        print(f"   Errors: {self.session_data['summary']['errors_count']}")
        print(f"   Logs Saved:")
        print(f"     JSON: {self.session_log}")
        print(f"     CSV:  {self.decisions_log}")

def create_decision_summary_report(log_dir="PRODUCTION/logs/decisions"):
    """Create summary report of all trading decisions"""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print("No decision logs found")
        return
        
    # Find all session files
    session_files = list(log_path.glob("session_*.json"))
    
    if not session_files:
        print("No session logs found")
        return
        
    print(f"\n📈 DECISION SUMMARY REPORT")
    print("=" * 60)
    
    for session_file in sorted(session_files)[-5:]:  # Last 5 sessions
        try:
            with open(session_file, 'r') as f:
                session = json.load(f)
                
            print(f"\n🕐 Session: {session['session_id']}")
            print(f"   Start: {session['start_time']}")
            print(f"   Liquidations: {len(session.get('liquidations', []))}")
            print(f"   Orders: {len(session.get('orders', []))}")
            print(f"   Errors: {len(session.get('errors', []))}")
            
            # Show recent orders
            orders = session.get('orders', [])
            if orders:
                print(f"   Recent Orders:")
                for order in orders[-3:]:  # Last 3 orders
                    print(f"     {order['symbol']} {order['side']} {order['quantity']} @ ${order['price']:.2f}")
                    
        except Exception as e:
            print(f"   Error reading {session_file}: {e}")

if __name__ == "__main__":
    create_decision_summary_report()