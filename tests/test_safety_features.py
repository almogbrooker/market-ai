#!/usr/bin/env python3
"""
Comprehensive tests for bot safety features
"""

import pytest
import json
import os
import tempfile
import uuid
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestSafetyFeatures:
    """Test bot safety and reliability features"""
    
    @pytest.fixture
    def mock_bot(self):
        """Create a mock bot instance for testing"""
        with patch('final_production_bot.tradeapi'), \
             patch('final_production_bot.os.getenv') as mock_getenv:
            
            mock_getenv.side_effect = lambda key: {
                'ALPACA_API_KEY': 'test_key',
                'ALPACA_SECRET_KEY': 'test_secret'
            }.get(key)
            
            from final_production_bot import FinalProductionBot
            
            # Create temporary state file
            temp_dir = tempfile.mkdtemp()
            bot = FinalProductionBot(paper=True)
            bot.state_file = os.path.join(temp_dir, 'test_bot_state.json')
            bot.api = Mock()
            
            yield bot
            
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_idempotent_order_id_generation(self, mock_bot):
        """Test that order IDs are unique and idempotent"""
        # Generate multiple order IDs
        order_ids = []
        for _ in range(100):
            order_id = mock_bot.generate_order_id('AAPL', 'buy')
            order_ids.append(order_id)
        
        # All should be unique
        assert len(set(order_ids)) == 100, "Order IDs should be unique"
        
        # Should contain symbol and side
        for order_id in order_ids[:5]:
            assert 'AAPL' in order_id, "Order ID should contain symbol"
            assert 'buy' in order_id, "Order ID should contain side"
            assert len(order_id) > 20, "Order ID should be sufficiently long"
    
    def test_daily_kill_switch_activation(self, mock_bot):
        """Test daily kill switch triggers at -3% loss"""
        # Mock account with 3.5% loss
        mock_account = Mock()
        mock_account.portfolio_value = '96500.00'  # Down from $100k
        mock_bot.api.get_account.return_value = mock_account
        
        # Set initial daily value
        mock_bot.state = {
            'daily_start_value': 100000.0,
            'last_reset_date': str(datetime.now().date()),
            'kill_switch_active': False
        }
        
        # Should activate kill switch
        result = mock_bot.check_daily_kill_switch()
        assert result is True, "Kill switch should activate at -3.5% loss"
        assert mock_bot.state['kill_switch_active'] is True
    
    def test_daily_kill_switch_no_activation(self, mock_bot):
        """Test kill switch doesn't activate with small losses"""
        # Mock account with 2% loss (below threshold)
        mock_account = Mock()
        mock_account.portfolio_value = '98000.00'  # Down from $100k
        mock_bot.api.get_account.return_value = mock_account
        
        # Set initial daily value
        mock_bot.state = {
            'daily_start_value': 100000.0,
            'last_reset_date': str(datetime.now().date()),
            'kill_switch_active': False
        }
        
        # Should NOT activate kill switch
        result = mock_bot.check_daily_kill_switch()
        assert result is False, "Kill switch should not activate at -2% loss"
        assert mock_bot.state['kill_switch_active'] is False
    
    def test_state_persistence(self, mock_bot):
        """Test state save/load functionality"""
        # Create test state
        test_state = {
            'open_orders': {'order1': {'symbol': 'AAPL', 'qty': 100}},
            'last_targets': {'AAPL': 0.25},
            'kill_switch_active': True,
            'test_field': 'test_value'
        }
        
        # Save state
        mock_bot.state = test_state
        mock_bot.save_state()
        
        # Verify file exists
        assert os.path.exists(mock_bot.state_file), "State file should be created"
        
        # Load state in new instance
        mock_bot.state = {}
        mock_bot.load_state()
        
        # Verify state loaded correctly
        assert mock_bot.state['test_field'] == 'test_value'
        assert mock_bot.state['kill_switch_active'] is True
        assert 'AAPL' in mock_bot.state['open_orders']['order1']['symbol']
    
    def test_429_rate_limit_backoff(self, mock_bot):
        """Test exponential backoff on 429 errors"""
        # Mock API to raise 429 error twice, then succeed
        mock_order = Mock()
        mock_order.id = 'test_order_123'
        
        mock_bot.api.submit_order.side_effect = [
            Exception("429 Rate limit exceeded"),
            Exception("429 Too many requests"),
            mock_order  # Success on third try
        ]
        
        with patch('time.sleep') as mock_sleep:
            result = mock_bot.submit_order_with_retry('AAPL', 100, 'buy')
        
        # Should succeed after retries
        assert result is not None, "Order should succeed after retries"
        assert result.id == 'test_order_123'
        
        # Should have called sleep with exponential backoff
        expected_sleeps = [1, 2]  # 1s, 2s backoff
        actual_sleeps = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_sleeps == expected_sleeps, "Should use exponential backoff"
    
    def test_partial_fill_handling(self, mock_bot):
        """Test partial fill detection and re-targeting"""
        # Mock partially filled order
        mock_order = Mock()
        mock_order.status = 'partially_filled'
        mock_order.filled_qty = '50'  # Half filled
        mock_order.id = 'original_order_123'
        
        # Mock new order for remainder
        mock_new_order = Mock()
        mock_new_order.id = 'new_order_456'
        
        mock_bot.api.get_order.return_value = mock_order
        mock_bot.api.cancel_order.return_value = Mock()
        mock_bot.api.submit_order.return_value = mock_new_order
        
        # Set up open order state
        mock_bot.state = {
            'open_orders': {
                'test_client_id': {
                    'symbol': 'AAPL',
                    'qty': 100,  # Originally wanted 100 shares
                    'side': 'buy',
                    'order_id': 'original_order_123',
                    'timestamp': str(datetime.now())
                }
            }
        }
        
        # Handle partial fills
        mock_bot.handle_partial_fills()
        
        # Should cancel original and submit new order
        mock_bot.api.cancel_order.assert_called_once_with('original_order_123')
        
        # Should update state with new order
        assert mock_bot.state['open_orders']['test_client_id']['order_id'] == 'new_order_456'
        assert mock_bot.state['open_orders']['test_client_id']['qty'] == 50  # Remaining quantity
    
    def test_order_state_tracking(self, mock_bot):
        """Test order tracking in state"""
        mock_order = Mock()
        mock_order.id = 'alpaca_order_123'
        mock_bot.api.submit_order.return_value = mock_order
        
        # Submit order
        result = mock_bot.submit_order_with_retry('AAPL', 100, 'buy')
        
        # Should track order in state
        assert result is not None
        order_states = mock_bot.state.get('open_orders', {})
        assert len(order_states) == 1
        
        # Check order details
        client_id = list(order_states.keys())[0]
        order_info = order_states[client_id]
        assert order_info['symbol'] == 'AAPL'
        assert order_info['qty'] == 100
        assert order_info['side'] == 'buy'
        assert order_info['order_id'] == 'alpaca_order_123'
    
    def test_error_recovery(self, mock_bot):
        """Test graceful error handling"""
        # Test state loading with corrupted file
        with open(mock_bot.state_file, 'w') as f:
            f.write("invalid json content")
        
        # Should handle gracefully
        mock_bot.load_state()
        assert isinstance(mock_bot.state, dict), "Should create default state on error"
        assert 'open_orders' in mock_bot.state
    
    def test_demo_mode_safety(self, mock_bot):
        """Test that demo mode doesn't make real API calls"""
        mock_bot.demo_mode = True
        
        # Kill switch check should return False in demo mode
        result = mock_bot.check_daily_kill_switch()
        assert result is False, "Demo mode should not activate kill switch"
        
        # Partial fill handling should exit early in demo mode
        mock_bot.handle_partial_fills()  # Should not raise errors
        
        # No API calls should be made
        mock_bot.api.get_account.assert_not_called()

class TestRebalancer:
    """Test the target-weights rebalancer"""
    
    @pytest.fixture
    def rebalancer(self):
        """Create rebalancer instance"""
        from src.trading.rebalance import TargetWeightsRebalancer
        return TargetWeightsRebalancer(max_gross_exposure=0.95)
    
    def test_rebalance_calculation(self, rebalancer):
        """Test basic rebalancing calculation"""
        target_weights = {'AAPL': 0.5, 'MSFT': -0.3}  # 50% long, 30% short
        current_positions = {'AAPL': 0, 'MSFT': 100}  # No AAPL, long MSFT
        current_prices = {'AAPL': 200.0, 'MSFT': 300.0}
        portfolio_value = 100000.0
        
        orders = rebalancer.calculate_rebalance_orders(
            target_weights, current_positions, current_prices, portfolio_value
        )
        
        assert len(orders) > 0, "Should generate rebalancing orders"
        
        # Should want to buy AAPL (currently 0, target 50%)
        aapl_order = next((o for o in orders if o.symbol == 'AAPL'), None)
        assert aapl_order is not None, "Should have AAPL order"
        assert aapl_order.side == 'buy', "Should buy AAPL"
        assert aapl_order.target_weight == 0.5
    
    def test_exposure_limiting(self, rebalancer):
        """Test that excessive exposure is scaled down"""
        # Target weights sum to 150% gross exposure
        target_weights = {'AAPL': 0.8, 'MSFT': -0.7}  # 150% gross
        current_positions = {}
        current_prices = {'AAPL': 200.0, 'MSFT': 300.0}
        portfolio_value = 100000.0
        
        orders = rebalancer.calculate_rebalance_orders(
            target_weights, current_positions, current_prices, portfolio_value
        )
        
        # Should scale down to fit max exposure (95%)
        total_target_exposure = sum(abs(o.target_weight) for o in orders)
        assert total_target_exposure <= 0.95, "Should limit total exposure"
    
    def test_minimum_trade_value(self, rebalancer):
        """Test that tiny trades are filtered out"""
        # Set very small position difference
        target_weights = {'AAPL': 0.001}  # 0.1% target
        current_positions = {'AAPL': 0}
        current_prices = {'AAPL': 200.0}
        portfolio_value = 10000.0  # Small portfolio
        
        orders = rebalancer.calculate_rebalance_orders(
            target_weights, current_positions, current_prices, portfolio_value
        )
        
        # Should filter out trades below minimum value ($100)
        small_orders = [o for o in orders if abs(o.share_diff * current_prices[o.symbol]) < 100]
        assert len(small_orders) == 0, "Should filter out small trades"
    
    def test_position_closing(self, rebalancer):
        """Test closing positions not in target weights"""
        target_weights = {'AAPL': 0.5}  # Only want AAPL
        current_positions = {'AAPL': 100, 'MSFT': 50, 'GOOGL': 25}  # Have others too
        current_prices = {'AAPL': 200.0, 'MSFT': 300.0, 'GOOGL': 150.0}
        portfolio_value = 100000.0
        
        orders = rebalancer.calculate_rebalance_orders(
            target_weights, current_positions, current_prices, portfolio_value
        )
        
        # Should have orders to close MSFT and GOOGL
        close_orders = [o for o in orders if o.target_weight == 0.0]
        symbols_to_close = {o.symbol for o in close_orders}
        assert 'MSFT' in symbols_to_close, "Should close MSFT position"
        assert 'GOOGL' in symbols_to_close, "Should close GOOGL position"

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])