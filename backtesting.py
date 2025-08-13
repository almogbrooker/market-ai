import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a single trade"""
    ticker: str
    quantity: float
    price: float
    timestamp: datetime
    side: str  # 'buy' or 'sell'
    commission: float = 0.0
    slippage: float = 0.0
    
    @property
    def value(self) -> float:
        return abs(self.quantity * self.price)
    
    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage

@dataclass
class Position:
    """Represents a position in a security"""
    ticker: str
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_price
    
    def update_price(self, current_price: float):
        """Update unrealized P&L with current price"""
        if self.quantity != 0:
            self.unrealized_pnl = (current_price - self.avg_price) * self.quantity

@dataclass
class BacktestResult:
    """Container for backtest results"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    
    # Transaction costs
    total_commission: float = 0.0
    total_slippage: float = 0.0
    turnover: float = 0.0
    
    # Additional metrics
    num_trades: int = 0
    avg_trade_duration: float = 0.0
    largest_loss: float = 0.0
    largest_gain: float = 0.0
    
    trades: List[Trade] = field(default_factory=list)
    portfolio_values: pd.Series = field(default_factory=pd.Series)
    positions_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    
class TransactionCostModel(ABC):
    """Abstract base class for transaction cost models"""
    
    @abstractmethod
    def calculate_cost(self, trade: Trade, market_data: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate commission and slippage for a trade"""
        pass

class FixedTransactionCost(TransactionCostModel):
    """Fixed commission and proportional slippage model"""
    
    def __init__(self, commission_rate: float = 0.001, slippage_rate: float = 0.001):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
    
    def calculate_cost(self, trade: Trade, market_data: Dict[str, Any]) -> Tuple[float, float]:
        commission = trade.value * self.commission_rate
        slippage = trade.value * self.slippage_rate
        return commission, slippage

class AdaptiveTransactionCost(TransactionCostModel):
    """Adaptive transaction cost model based on volume and volatility"""
    
    def __init__(self, base_commission: float = 0.0005, base_slippage: float = 0.0005):
        self.base_commission = base_commission
        self.base_slippage = base_slippage
    
    def calculate_cost(self, trade: Trade, market_data: Dict[str, Any]) -> Tuple[float, float]:
        # Get market data
        volume = market_data.get('volume', 1000000)
        volatility = market_data.get('volatility', 0.02)
        
        # Adaptive commission (higher for low volume stocks)
        volume_factor = max(1.0, 1000000 / volume)  # Scale by average daily volume
        commission_rate = self.base_commission * volume_factor
        commission = trade.value * commission_rate
        
        # Adaptive slippage (higher for volatile stocks and larger trades)
        volatility_factor = max(1.0, volatility / 0.02)
        trade_size_factor = max(1.0, trade.value / 100000)  # Scale by trade size
        slippage_rate = self.base_slippage * volatility_factor * trade_size_factor
        slippage = trade.value * slippage_rate
        
        return commission, slippage

class RiskManager:
    """Risk management for trading strategies"""
    
    def __init__(self, max_position_size: float = 0.1, max_portfolio_risk: float = 0.02,
                 stop_loss_pct: float = 0.05, take_profit_pct: float = 0.15):
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk  
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
    
    def check_position_size(self, position_value: float, portfolio_value: float) -> bool:
        """Check if position size is within limits"""
        position_weight = position_value / (portfolio_value + 1e-8)
        return position_weight <= self.max_position_size
    
    def check_portfolio_risk(self, portfolio_var: float) -> bool:
        """Check if portfolio risk is within limits"""
        return portfolio_var <= self.max_portfolio_risk
    
    def apply_stop_loss_take_profit(self, positions: Dict[str, Position], 
                                   current_prices: Dict[str, float]) -> List[Trade]:
        """Generate stop-loss or take-profit trades"""
        trades = []
        
        for ticker, position in positions.items():
            if position.quantity == 0:
                continue
                
            current_price = current_prices.get(ticker, position.avg_price)
            pnl_pct = (current_price - position.avg_price) / position.avg_price
            
            # Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                trade = Trade(
                    ticker=ticker,
                    quantity=-position.quantity,  # Close position
                    price=current_price,
                    timestamp=datetime.now(),
                    side='sell' if position.quantity > 0 else 'buy'
                )
                trades.append(trade)
                logger.info(f"Stop-loss triggered for {ticker}: {pnl_pct:.2%} loss")
            
            # Take profit
            elif pnl_pct >= self.take_profit_pct:
                trade = Trade(
                    ticker=ticker,
                    quantity=-position.quantity,  # Close position
                    price=current_price,
                    timestamp=datetime.now(),
                    side='sell' if position.quantity > 0 else 'buy'
                )
                trades.append(trade)
                logger.info(f"Take-profit triggered for {ticker}: {pnl_pct:.2%} gain")
        
        return trades

class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, initial_capital: float = 100000,
                 transaction_cost_model: Optional[TransactionCostModel] = None,
                 risk_manager: Optional[RiskManager] = None,
                 benchmark: Optional[str] = None):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_cost_model = transaction_cost_model or FixedTransactionCost()
        self.risk_manager = risk_manager or RiskManager()
        self.benchmark = benchmark
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_values: List[float] = [initial_capital]
        self.timestamps: List[datetime] = []
        self.cash = initial_capital
        
    def execute_trade(self, trade: Trade, market_data: Dict[str, Any]):
        """Execute a trade and update positions"""
        try:
            # Calculate transaction costs
            commission, slippage = self.transaction_cost_model.calculate_cost(trade, market_data)
            trade.commission = commission
            trade.slippage = slippage
            
            # Check if we have enough cash for buy orders
            trade_cost = trade.value + commission + slippage
            if trade.quantity > 0 and trade_cost > self.cash:
                logger.warning(f"Insufficient cash for trade: {trade.ticker}, required: {trade_cost:.2f}, available: {self.cash:.2f}")
                return False
            
            # Update position
            if trade.ticker not in self.positions:
                self.positions[trade.ticker] = Position(ticker=trade.ticker)
            
            position = self.positions[trade.ticker]
            
            if trade.quantity > 0:  # Buy
                # Update average price
                total_quantity = position.quantity + trade.quantity
                if total_quantity > 0:
                    position.avg_price = ((position.avg_price * position.quantity + 
                                         trade.price * trade.quantity) / total_quantity)
                position.quantity = total_quantity
                self.cash -= trade_cost
                
            else:  # Sell
                # Realize P&L
                realized_pnl = -trade.quantity * (trade.price - position.avg_price)
                position.realized_pnl += realized_pnl
                position.quantity += trade.quantity  # trade.quantity is negative for sells
                
                self.cash += abs(trade.quantity * trade.price) - commission - slippage
                
                # If position is closed, reset average price
                if abs(position.quantity) < 1e-8:
                    position.quantity = 0.0
                    position.avg_price = 0.0
            
            self.trades.append(trade)
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def update_portfolio_value(self, current_prices: Dict[str, float], timestamp: datetime):
        """Update portfolio value with current market prices"""
        try:
            total_value = self.cash
            
            # Update positions and calculate market value
            for ticker, position in self.positions.items():
                if position.quantity != 0:
                    current_price = current_prices.get(ticker, position.avg_price)
                    position.update_price(current_price)
                    total_value += position.quantity * current_price
            
            self.portfolio_values.append(total_value)
            self.timestamps.append(timestamp)
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    def run_backtest(self, data: pd.DataFrame, strategy_func: Callable, 
                    **strategy_params) -> BacktestResult:
        """Run the backtest with given strategy"""
        try:
            logger.info("Starting backtest...")
            
            for idx, row in data.iterrows():
                timestamp = row.name if isinstance(row.name, datetime) else idx
                
                # Prepare market data for current period
                market_data = row.to_dict()
                current_prices = {col.split('_')[0]: row[col] for col in data.columns 
                                if col.endswith('_Close')}
                
                if not current_prices:
                    # Fallback: assume data has 'Close' column for single asset
                    if 'Close' in row:
                        current_prices = {'ASSET': row['Close']}
                
                # Generate trading signals
                signals = strategy_func(data.iloc[:idx+1], self.positions, **strategy_params)
                
                # Apply risk management
                risk_trades = self.risk_manager.apply_stop_loss_take_profit(
                    self.positions, current_prices
                )
                all_trades = signals + risk_trades
                
                # Execute trades
                for trade in all_trades:
                    trade.timestamp = timestamp
                    success = self.execute_trade(trade, market_data)
                    if not success:
                        logger.warning(f"Failed to execute trade: {trade}")
                
                # Update portfolio value
                self.update_portfolio_value(current_prices, timestamp)
            
            # Calculate final results
            result = self._calculate_results(data)
            
            logger.info("Backtest completed successfully")
            logger.info(f"Total return: {result.total_return:.2%}")
            logger.info(f"Sharpe ratio: {result.sharpe_ratio:.3f}")
            logger.info(f"Max drawdown: {result.max_drawdown:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        try:
            if len(self.portfolio_values) < 2:
                return BacktestResult()
            
            # Convert to pandas series for easier calculation
            portfolio_series = pd.Series(self.portfolio_values, index=self.timestamps)
            returns = portfolio_series.pct_change().dropna()
            
            # Basic metrics
            total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (annualized_return - 0.02) / (volatility + 1e-8)  # Assuming 2% risk-free rate
            
            # Downside metrics
            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_return - 0.02) / (downside_vol + 1e-8)
            
            # Drawdown metrics
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            calmar_ratio = annualized_return / (abs(max_drawdown) + 1e-8)
            
            # Risk metrics
            var_95 = returns.quantile(0.05)
            expected_shortfall = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
            
            # Trading metrics
            if self.trades:
                total_commission = sum(trade.commission for trade in self.trades)
                total_slippage = sum(trade.slippage for trade in self.trades)
                
                # Win rate calculation
                profitable_trades = [t for t in self.trades if self._get_trade_pnl(t) > 0]
                win_rate = len(profitable_trades) / len(self.trades)
                
                # Profit factor
                gross_profit = sum(max(0, self._get_trade_pnl(t)) for t in self.trades)
                gross_loss = abs(sum(min(0, self._get_trade_pnl(t)) for t in self.trades))
                profit_factor = gross_profit / (gross_loss + 1e-8)
            else:
                total_commission = total_slippage = win_rate = profit_factor = 0
            
            # Turnover
            total_traded_value = sum(trade.value for trade in self.trades)
            avg_portfolio_value = np.mean(self.portfolio_values)
            turnover = total_traded_value / (avg_portfolio_value + 1e-8)
            
            result = BacktestResult(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                total_commission=total_commission,
                total_slippage=total_slippage,
                turnover=turnover,
                num_trades=len(self.trades),
                trades=self.trades.copy(),
                portfolio_values=portfolio_series
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating results: {e}")
            return BacktestResult()
    
    def _get_trade_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a single trade (simplified)"""
        # This is a simplified calculation - in reality, you'd need to track
        # the full position lifecycle
        return 0.0

# Example strategy functions
def buy_and_hold_strategy(data: pd.DataFrame, positions: Dict[str, Position], 
                         **kwargs) -> List[Trade]:
    """Simple buy and hold strategy"""
    trades = []
    
    # Only buy on the first day if we don't have positions
    if len(data) == 1 and not positions:
        if 'Close' in data.columns:
            price = data['Close'].iloc[-1]
            trade = Trade(
                ticker='ASSET',
                quantity=1000,  # Fixed quantity
                price=price,
                timestamp=datetime.now(),
                side='buy'
            )
            trades.append(trade)
    
    return trades

def mean_reversion_strategy(data: pd.DataFrame, positions: Dict[str, Position],
                           lookback: int = 20, threshold: float = 2.0) -> List[Trade]:
    """Simple mean reversion strategy"""
    trades = []
    
    if len(data) < lookback + 1:
        return trades
    
    if 'Close' not in data.columns:
        return trades
    
    # Calculate rolling statistics
    prices = data['Close']
    rolling_mean = prices.rolling(lookback).mean()
    rolling_std = prices.rolling(lookback).std()
    
    current_price = prices.iloc[-1]
    current_mean = rolling_mean.iloc[-1]
    current_std = rolling_std.iloc[-1]
    
    if pd.isna(current_mean) or pd.isna(current_std):
        return trades
    
    # Z-score
    z_score = (current_price - current_mean) / (current_std + 1e-8)
    
    # Trading logic
    if z_score > threshold and 'ASSET' in positions and positions['ASSET'].quantity > 0:
        # Price is high, sell
        trade = Trade(
            ticker='ASSET',
            quantity=-positions['ASSET'].quantity,
            price=current_price,
            timestamp=datetime.now(),
            side='sell'
        )
        trades.append(trade)
    elif z_score < -threshold:
        # Price is low, buy
        quantity = 1000  # Fixed quantity
        if 'ASSET' not in positions or positions['ASSET'].quantity <= 0:
            trade = Trade(
                ticker='ASSET',
                quantity=quantity,
                price=current_price,
                timestamp=datetime.now(),
                side='buy'
            )
            trades.append(trade)
    
    return trades

def run_sample_backtest(data: pd.DataFrame) -> BacktestResult:
    """Run a sample backtest with mean reversion strategy"""
    try:
        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=100000,
            transaction_cost_model=FixedTransactionCost(commission_rate=0.001, slippage_rate=0.0005),
            risk_manager=RiskManager(stop_loss_pct=0.1, take_profit_pct=0.2)
        )
        
        # Run backtest
        result = engine.run_backtest(
            data=data,
            strategy_func=mean_reversion_strategy,
            lookback=20,
            threshold=1.5
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error running sample backtest: {e}")
        raise