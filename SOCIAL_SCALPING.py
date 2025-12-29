"""
🔥 BLADE RUNNER SYMPHONY ULTIMATE V2.0 - LE SIPHONNEUR QUANTIQUE SOCIAL
=======================================================================

🎵 "Le marché est une symphonie de classes en guerre.
    Chaque minute, chaque seconde, chaque tick est une note.
    Les riches orchestrent, les pauvres se révoltent,
    les vampires sucent, les traîtres trahissent.
    Nous écoutons, nous apprenons, nous nous adaptons.
    Nous scalpons 1, 2, 3, 4, 5 fois par minute.
    Chaque trade est une leçon, chaque profit est une victoire.
    L'apprentissage est infini, l'évolution est permanente.
    Nous commençons modestes, nous finissons en géants.
    Le capital des riches devient notre capital.
    La peur des pauvres devient notre force.
    Le temps est notre allié, la patience notre arme.
    Nous siphonnerons jusqu'à ce que le dernier dollar change de main." 🎵

SYSTÈME DE SCALPING QUANTIQUE SOCIAL À APPRENTISSAGE AUTOMATIQUE
- Multi-scalping : 2-6 trades par minute selon les conditions
- Apprentissage profond : Chaque trade améliore le système
- Évolution dynamique : Les paramètres s'adaptent en temps réel
- Conscience de classe : Traque spécifique des riches/vampires
- Généalogie avancée : 7 générations de bougies analysées
- Mémoire holographique : Tous les trades sont mémorisés
- Optimisation continue : Les stratégies évoluent chaque heure
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import json
import os
import time
import threading
import traceback
import hashlib
import pickle
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Deque
from enum import Enum
from collections import deque, defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import talib
from scipy import fft, signal
from scipy.signal import hilbert, find_peaks, savgol_filter, butter, filtfilt
from scipy.stats import skew, kurtosis, entropy, linregress
from scipy.optimize import minimize, differential_evolution
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers, models
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION NUCLÉAIRE - LE CŒUR DE LA BÊTE
# ============================================================================

@dataclass
class QuantumConfig:
    """CONFIGURATION QUANTIQUE DU SIPHONNEUR"""
    
    # IDENTITÉ DU SIPHONNEUR
    version: str = "2.0.0"
    name: str = "SYMPHONY_ULTIMATE"
    creator: str = "MATHIEU_ANGEL"
    
    # CONNEXION MT5
    login: int = 10008836538
    password: str = "Mh-vYv0b"
    server: str = "MetaQuotes-Demo"
    symbol: str = "XAUUSD"
    
    # HORLOGE QUANTIQUE (en secondes)
    tick_frequency: float = 0.001          # 1ms - rythme cardiaque
    analysis_frequency: float = 0.01       # 10ms - pensée rapide
    decision_frequency: float = 0.1        # 100ms - prise de décision
    scalp_frequency: float = 0.5           # 500ms - fréquence de scalp
    
    # MULTI-SCALPING AGGRESSIF
    scalp_per_minute_target: int = 4       # Objectif : 4 scalps/minute
    scalp_opportunities: List[int] = field(default_factory=lambda: [3, 8, 15, 25, 40, 55])  # Secondes dans la minute
    max_concurrent_scalps: int = 3         # Maximum de scalps simultanés
    scalp_duration_max: int = 45           # Durée max d'un scalp (secondes)
    
    # GESTION DE CAPITAL ÉVOLUTIVE
    base_capital: float = 10000.0          # Capital de base
    risk_per_trade_base: float = 0.5       # 0.5% de risque par trade
    risk_multiplier_learning: float = 1.0  # Multiplicateur basé sur l'apprentissage
    lot_base: float = 0.01                 # Lot de base
    lot_growth_factor: float = 1.1         # Croissance des lots après succès
    max_lot_size: float = 0.50             # Lot maximum
    compound_profits: bool = True          # Capitalisation des profits
    
    # SEUILS DYNAMIQUES (seront ajustés par l'IA)
    profit_target_initial: float = 2.0     # 2$ cible initiale
    stop_loss_initial: float = -4.0        # -4$ stop initial
    trailing_stop_activate: float = 1.5    # Active trailing à 1.5$
    trailing_stop_distance: float = 0.5    # Distance du trailing
    
    # APPRENTISSAGE PROFOND
    learning_enabled: bool = True          # Apprentissage activé
    memory_size: int = 10000               # Taille de la mémoire
    replay_batch_size: int = 64            # Taille des batchs d'apprentissage
    learning_interval: int = 60            # Apprend toutes les 60 secondes
    model_save_interval: int = 300         # Sauvegarde modèle toutes les 5 minutes
    
    # RÉSEAUX DE NEURONES
    neural_layers: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    training_epochs: int = 10
    
    # CLASSIFICATION SOCIALE AVANCÉE
    social_classes: List[str] = field(default_factory=lambda: [
        "PAUVRE_REVOLTE", 
        "PAUVRE_SOUMIS",
        "CLASSE_MOYENNE_STABLE",
        "CLASSE_MOYENNE_ANXIEUSE",
        "RICHE_CALME",
        "RICHE_AGITE",
        "VAMPIRE_SANG_FROID",
        "VAMPIRE_AFFAME",
        "TRAITRE_CALCULATEUR",
        "TRAITRE_PANIQUE"
    ])
    
    # ARCHIVES ET SURVEILLANCE
    data_directory: str = "symphony_archives_v2"
    trade_log_file: str = "trades_complete.json"
    learning_log_file: str = "learning_evolution.json"
    performance_file: str = "performance_metrics.json"
    
    # MODES D'OPÉRATION
    operational_mode: str = "AGGRESSIVE_LEARNING"  # AGGRESSIVE_LEARNING, CONSERVATIVE, HYPER_AGGRESSIVE
    learning_phase: str = "ACQUISITION"  # ACQUISITION, CONSOLIDATION, EXPLOITATION
    
    # SYSTÈME DE SANTÉ
    health_check_interval: int = 30        # Vérification santé toutes les 30s
    auto_recovery: bool = True             # Récupération automatique
    max_errors_before_reset: int = 10      # Réinitialisation après 10 erreurs
    
    # JOURNALISATION DÉTAILLÉE
    log_level: str = "DEBUG"               # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True
    log_to_console: bool = True

# ============================================================================
# STRUCTURES DE DONNÉES QUANTIQUES
# ============================================================================

class QuantumState(Enum):
    """ÉTATS QUANTIQUES DU SYSTÈME"""
    SUPERPOSITION = "Superposition"        # Indécision quantique
    ENTANGLEMENT = "Intrication"           # Corrélation avec le marché
    COLLAPSE_BUY = "Effondrement_ACHAT"    # Décision d'achat
    COLLAPSE_SELL = "Effondrement_VENTE"   # Décision de vente
    TUNNELING = "Effet_Tunnel"             # Percée de niveau
    COHERENCE = "Cohérence"                # Direction claire
    DECOHERENCE = "Décohérence"           # Chaos et incertitude
    REVOLUTION = "Révolution"              Changement de régime

@dataclass
class CandleGenome:
    """GÉNOME COMPLET D'UNE BOUGIE - ADN SOCIAL QUANTIQUE"""
    
    # IDENTITÉ UNIQUE
    genome_id: str
    timestamp: datetime
    generation: int
    
    # CARACTÉRISTIQUES PHYSIQUES
    open: float
    high: float
    low: float
    close: float
    volume: float
    body_size: float
    wick_upper: float
    wick_lower: float
    body_position: float  # 0-1 dans la range
    
    # SIGNATURE QUANTIQUE
    quantum_signature: str  # Hash quantique
    frequency_dominant: float
    amplitude_quantum: float
    phase_coherence: float
    energy_spectral: float
    
    # CLASSIFICATION SOCIALE
    social_class: str
    social_mobility: float  # -100 à +100
    class_stability: float  # 0-1
    revolt_potential: float  # 0-1
    
    # PERSONNALITÉ MBTI ÉTENDUE
    mbti_type: str
    confidence_score: float
    aggression_level: float
    predictability: float
    
    # GÉNÉALOGIE
    parent_genomes: List[str]
    child_genomes: List[str]
    family_lineage: str
    genetic_mutations: List[str]
    
    # APPRENTISSAGE
    trade_outcomes: List[Dict] = field(default_factory=list)
    learning_weight: float = 1.0
    adaptation_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour sérialisation"""
        return {
            'genome_id': self.genome_id,
            'timestamp': self.timestamp.isoformat(),
            'generation': self.generation,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'body_size': self.body_size,
            'wick_upper': self.wick_upper,
            'wick_lower': self.wick_lower,
            'body_position': self.body_position,
            'quantum_signature': self.quantum_signature,
            'frequency_dominant': self.frequency_dominant,
            'amplitude_quantum': self.amplitude_quantum,
            'phase_coherence': self.phase_coherence,
            'energy_spectral': self.energy_spectral,
            'social_class': self.social_class,
            'social_mobility': self.social_mobility,
            'class_stability': self.class_stability,
            'revolt_potential': self.revolt_potential,
            'mbti_type': self.mbti_type,
            'confidence_score': self.confidence_score,
            'aggression_level': self.aggression_level,
            'predictability': self.predictability,
            'parent_genomes': self.parent_genomes,
            'child_genomes': self.child_genomes,
            'family_lineage': self.family_lineage,
            'genetic_mutations': self.genetic_mutations,
            'trade_outcomes': self.trade_outcomes,
            'learning_weight': self.learning_weight,
            'adaptation_score': self.adaptation_score
        }

@dataclass
class QuantumTrade:
    """TRADE QUANTIQUE COMPLET - MÉMOIRE HOLOGRAPHIQUE"""
    
    # IDENTIFICATION
    trade_id: str
    open_time: datetime
    close_time: Optional[datetime] = None
    
    # EXÉCUTION
    direction: str  # BUY/SELL
    entry_price: float
    exit_price: Optional[float] = None
    lot_size: float
    commission: float = 0.0
    swap: float = 0.0
    
    # GESTION
    stop_loss: float
    take_profit: float
    trailing_stop_activated: bool = False
    trailing_stop_price: Optional[float] = None
    
    # RÉSULTATS
    profit: Optional[float] = None
    profit_pips: Optional[float] = None
    profit_percentage: Optional[float] = None
    duration_seconds: Optional[float] = None
    
    # CONTEXTE QUANTIQUE
    quantum_state_at_entry: str
    quantum_state_at_exit: Optional[str] = None
    social_class_targeted: str
    genome_id: str
    
    # DÉCISION
    decision_confidence: float
    decision_reason: str
    decision_algorithm: str
    risk_score: float
    
    # APPRENTISSAGE
    learning_features: Dict[str, Any] = field(default_factory=dict)
    outcome_classification: str = ""  # WIN, LOSS, BREAK_EVEN
    lesson_learned: str = ""
    adaptation_applied: bool = False
    
    # MÉTADONNÉES
    scalp_number: int = 0  # Quelème scalp de la minute
    minute_of_day: int = 0
    market_regime: str = ""
    
    def calculate_results(self, current_price: float = None):
        """Calcule les résultats du trade"""
        if self.exit_price is not None:
            price_diff = self.exit_price - self.entry_price
            if self.direction == "SELL":
                price_diff = -price_diff
            
            self.profit = price_diff * self.lot_size * 100000  # Pour XAUUSD
            self.profit_pips = price_diff * 10000
            self.profit_percentage = (self.profit / (self.entry_price * self.lot_size * 100000)) * 100
            
            if self.close_time:
                self.duration_seconds = (self.close_time - self.open_time).total_seconds()
        
        elif current_price is not None:
            # Calcul en temps réel
            price_diff = current_price - self.entry_price
            if self.direction == "SELL":
                price_diff = -price_diff
            
            self.profit = price_diff * self.lot_size * 100000
            self.profit_pips = price_diff * 10000
            self.profit_percentage = (self.profit / (self.entry_price * self.lot_size * 100000)) * 100
            
            if self.open_time:
                self.duration_seconds = (datetime.now() - self.open_time).total_seconds()
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire"""
        self.calculate_results()
        
        return {
            'trade_id': self.trade_id,
            'open_time': self.open_time.isoformat(),
            'close_time': self.close_time.isoformat() if self.close_time else None,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'lot_size': self.lot_size,
            'commission': self.commission,
            'swap': self.swap,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop_activated': self.trailing_stop_activated,
            'trailing_stop_price': self.trailing_stop_price,
            'profit': self.profit,
            'profit_pips': self.profit_pips,
            'profit_percentage': self.profit_percentage,
            'duration_seconds': self.duration_seconds,
            'quantum_state_at_entry': self.quantum_state_at_entry,
            'quantum_state_at_exit': self.quantum_state_at_exit,
            'social_class_targeted': self.social_class_targeted,
            'genome_id': self.genome_id,
            'decision_confidence': self.decision_confidence,
            'decision_reason': self.decision_reason,
            'decision_algorithm': self.decision_algorithm,
            'risk_score': self.risk_score,
            'learning_features': self.learning_features,
            'outcome_classification': self.outcome_classification,
            'lesson_learned': self.lesson_learned,
            'adaptation_applied': self.adaptation_applied,
            'scalp_number': self.scalp_number,
            'minute_of_day': self.minute_of_day,
            'market_regime': self.market_regime
        }

@dataclass  
class LearningMemory:
    """MÉMOIRE D'APPRENTISSAGE HOLOGRAPHIQUE"""
    
    # STOCKAGE DES EXPÉRIENCES
    experiences: Deque[Dict] = field(default_factory=lambda: deque(maxlen=10000))
    
    # STATISTIQUES D'APPRENTISSAGE
    total_experiences: int = 0
    learning_cycles: int = 0
    adaptation_count: int = 0
    
    # PERFORMANCE ÉVOLUTIVE
    win_rate_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    profit_factor_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    avg_win_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    avg_loss_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    
    # CONNAISSANCE ACQUISE
    market_patterns: Dict[str, Dict] = field(default_factory=dict)
    social_class_behavior: Dict[str, Dict] = field(default_factory=dict)
    quantum_state_transitions: Dict[str, Dict] = field(default_factory=dict)
    
    # MODÈLES APPRENTIS
    decision_models: Dict[str, Any] = field(default_factory=dict)
    risk_models: Dict[str, Any] = field(default_factory=dict)
    timing_models: Dict[str, Any] = field(default_factory=dict)
    
    def add_experience(self, trade: QuantumTrade, market_context: Dict):
        """Ajoute une expérience d'apprentissage"""
        experience = {
            'timestamp': datetime.now().isoformat(),
            'trade_data': trade.to_dict(),
            'market_context': market_context,
            'learning_vector': self.create_learning_vector(trade, market_context)
        }
        
        self.experiences.append(experience)
        self.total_experiences += 1
        
        # Classification de l'expérience
        if trade.profit is not None:
            if trade.profit > 0:
                outcome = 'WIN'
                self.win_rate_history.append(1.0)
            else:
                outcome = 'LOSS'
                self.win_rate_history.append(0.0)
            
            trade.outcome_classification = outcome
    
    def create_learning_vector(self, trade: QuantumTrade, context: Dict) -> Dict:
        """Crée un vecteur d'apprentissage"""
        return {
            'features': {
                'social_class': trade.social_class_targeted,
                'quantum_state': trade.quantum_state_at_entry,
                'time_of_day': trade.open_time.hour * 60 + trade.open_time.minute,
                'scalp_number': trade.scalp_number,
                'decision_confidence': trade.decision_confidence,
                'risk_score': trade.risk_score,
                'market_regime': context.get('market_regime', ''),
                'volatility': context.get('volatility', 0),
                'volume_ratio': context.get('volume_ratio', 1.0),
                'trend_strength': context.get('trend_strength', 0)
            },
            'label': 1 if trade.profit and trade.profit > 0 else 0,
            'reward': trade.profit if trade.profit else 0
        }

# ============================================================================
# MOTEUR QUANTIQUE - CERVEAU DU SIPHONNEUR
# ============================================================================

class QuantumEngine:
    """MOTEUR D'ANALYSE QUANTIQUE - CERVEAU PRINCIPAL"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.initialize_engines()
        
    def initialize_engines(self):
        """Initialise tous les moteurs d'analyse"""
        
        # MOTEURS D'ANALYSE
        self.microstructure_analyzer = MicrostructureAnalyzer()
        self.wavelet_analyzer = WaveletAnalyzer()
        self.hilbert_analyzer = HilbertAnalyzer()
        self.fractal_analyzer = FractalAnalyzer()
        self.entropy_analyzer = EntropyAnalyzer()
        
        # CLASSIFICATEURS
        self.social_classifier = SocialClassifier(self.config.social_classes)
        self.personality_classifier = PersonalityClassifier()
        self.regime_classifier = MarketRegimeClassifier()
        
        # DÉTECTEURS
        self.anomaly_detector = AnomalyDetector()
        self.pattern_detector = PatternDetector()
        self.cycle_detector = CycleDetector()
        
        # PRÉDICTEURS
        self.momentum_predictor = MomentumPredictor()
        self.reversal_predictor = ReversalPredictor()
        self.volatility_predictor = VolatilityPredictor()
        
        # FILTRES
        self.kalman_filter = KalmanFilter()
        self.bandpass_filter = BandpassFilter()
        self.noise_filter = NoiseFilter()
        
    def analyze_tick_stream(self, tick_stream: List[Dict]) -> Dict:
        """Analyse un flux de ticks en temps réel"""
        
        if len(tick_stream) < 100:
            return {}
        
        # Extraction des données
        prices = [t['bid'] for t in tick_stream]
        times = [t['timestamp'] for t in tick_stream]
        volumes = [t.get('volume', 0) for t in tick_stream]
        
        # Analyse complète
        analysis = {
            'timestamp': datetime.now(),
            'price_current': prices[-1],
            'analyses': {}
        }
        
        # 1. ANALYSE MICROSTRUCTURELLE
        analysis['analyses']['microstructure'] = self.microstructure_analyzer.analyze(prices)
        
        # 2. TRANSFORMÉE EN ONDELETTES
        analysis['analyses']['wavelet'] = self.wavelet_analyzer.analyze(prices)
        
        # 3. TRANSFORMÉE DE HILBERT
        analysis['analyses']['hilbert'] = self.hilbert_analyzer.analyze(prices)
        
        # 4. ANALYSE FRACTALE
        analysis['analyses']['fractal'] = self.fractal_analyzer.analyze(prices)
        
        # 5. ANALYSE D'ENTROPIE
        analysis['analyses']['entropy'] = self.entropy_analyzer.analyze(prices)
        
        # 6. DÉTECTION D'ANOMALIES
        analysis['analyses']['anomalies'] = self.anomaly_detector.detect(prices)
        
        # 7. DÉTECTION DE PATTERNS
        analysis['analyses']['patterns'] = self.pattern_detector.detect(prices)
        
        # 8. DÉTECTION DE CYCLES
        analysis['analyses']['cycles'] = self.cycle_detector.detect(prices)
        
        # 9. CLASSIFICATION SOCIALE
        analysis['analyses']['social'] = self.social_classifier.classify(prices, volumes)
        
        # 10. PRÉDICTION DE MOMENTUM
        analysis['analyses']['momentum'] = self.momentum_predictor.predict(prices)
        
        # 11. PRÉDICTION DE REVERSAL
        analysis['analyses']['reversal'] = self.reversal_predictor.predict(prices)
        
        # 12. FILTRAGE KALMAN
        analysis['analyses']['kalman'] = self.kalman_filter.filter(prices)
        
        # Synthèse quantique
        analysis['quantum_synthesis'] = self.synthesize_quantum_state(analysis['analyses'])
        
        return analysis
    
    def synthesize_quantum_state(self, analyses: Dict) -> Dict:
        """Synthétise un état quantique à partir de toutes les analyses"""
        
        scores = {
            'trend_strength': 0.0,
            'volatility': 0.0,
            'predictability': 0.0,
            'energy': 0.0,
            'coherence': 0.0,
            'entropy': 0.0
        }
        
        # Agréger les scores
        if 'microstructure' in analyses:
            micro = analyses['microstructure']
            scores['volatility'] = micro.get('volatility', 0)
            scores['predictability'] = 1.0 - micro.get('entropy', 0.5)
        
        if 'wavelet' in analyses:
            wavelet = analyses['wavelet']
            scores['energy'] = wavelet.get('total_energy', 0)
        
        if 'hilbert' in analyses:
            hilbert = analyses['hilbert']
            scores['coherence'] = hilbert.get('coherence', 0)
        
        if 'entropy' in analyses:
            entropy = analyses['entropy']
            scores['entropy'] = entropy.get('shannon_entropy', 0.5)
        
        # Déterminer l'état quantique
        quantum_state = self.determine_quantum_state(scores)
        
        return {
            'scores': scores,
            'quantum_state': quantum_state,
            'state_confidence': self.calculate_state_confidence(scores),
            'transition_probability': self.calculate_transition_probability(scores)
        }
    
    def determine_quantum_state(self, scores: Dict) -> str:
        """Détermine l'état quantique dominant"""
        
        if scores['coherence'] > 0.7 and scores['entropy'] < 0.3:
            return QuantumState.COHERENCE.value
        
        elif scores['entropy'] > 0.7 and scores['predictability'] < 0.3:
            return QuantumState.DECOHERENCE.value
        
        elif scores['volatility'] > 0.6 and scores['energy'] > 0.5:
            return QuantumState.REVOLUTION.value
        
        elif scores['trend_strength'] > 0.6:
            if scores['predictability'] > 0.5:
                return QuantumState.COLLAPSE_BUY.value if np.random.random() > 0.5 else QuantumState.COLLAPSE_SELL.value
        
        elif scores['volatility'] < 0.2 and scores['energy'] < 0.3:
            return QuantumState.TUNNELING.value
        
        else:
            return QuantumState.SUPERPOSITION.value
    
    def calculate_state_confidence(self, scores: Dict) -> float:
        """Calcule la confiance dans l'état quantique"""
        confidence = 0.0
        weights = {
            'coherence': 0.3,
            'entropy': 0.25,
            'predictability': 0.2,
            'volatility': 0.15,
            'energy': 0.1
        }
        
        for key, weight in weights.items():
            if key in scores:
                confidence += scores[key] * weight
        
        return min(max(confidence, 0.0), 1.0)
    
    def calculate_transition_probability(self, scores: Dict) -> Dict:
        """Calcule les probabilités de transition entre états"""
        
        transitions = {}
        states = [state.value for state in QuantumState]
        
        for state in states:
            # Probabilité basée sur les scores actuels
            prob = np.random.random() * 0.3  # Base aléatoire
            
            if state == QuantumState.COHERENCE.value:
                prob += scores.get('coherence', 0) * 0.5
            
            elif state == QuantumState.DECOHERENCE.value:
                prob += scores.get('entropy', 0) * 0.5
            
            elif QuantumState.COLLAPSE_BUY.value in state or QuantumState.COLLAPSE_SELL.value in state:
                prob += scores.get('trend_strength', 0) * 0.3
            
            transitions[state] = min(max(prob, 0.0), 1.0)
        
        # Normaliser
        total = sum(transitions.values())
        if total > 0:
            for state in transitions:
                transitions[state] /= total
        
        return transitions

# ============================================================================
# ANALYSEURS SPÉCIALISÉS
# ============================================================================

class MicrostructureAnalyzer:
    """Analyse de la microstructure des prix"""
    
    def analyze(self, prices: List[float]) -> Dict:
        if len(prices) < 20:
            return {}
        
        prices_np = np.array(prices)
        returns = np.diff(prices_np) / prices_np[:-1]
        
        return {
            'volatility': float(np.std(returns) * 100),
            'skewness': float(skew(returns)) if len(returns) > 2 else 0,
            'kurtosis': float(kurtosis(returns)) if len(returns) > 3 else 0,
            'entropy': float(entropy(np.abs(returns) + 1e-10)),
            'autocorrelation': self.calculate_autocorrelation(returns),
            'hurst_exponent': self.calculate_hurst_exponent(prices_np),
            'variance_ratio': self.calculate_variance_ratio(returns)
        }
    
    def calculate_autocorrelation(self, returns: np.ndarray, lag: int = 5) -> float:
        if len(returns) < lag * 2:
            return 0.0
        return float(np.corrcoef(returns[:-lag], returns[lag:])[0, 1])
    
    def calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calcule l'exposant de Hurst pour la persistence"""
        if len(prices) < 100:
            return 0.5
        
        lags = range(2, 100)
        tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
        
        try:
            hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
            return float(hurst)
        except:
            return 0.5
    
    def calculate_variance_ratio(self, returns: np.ndarray) -> float:
        """Test de random walk"""
        if len(returns) < 20:
            return 1.0
        
        var_1 = np.var(returns)
        returns_2 = returns[:-1] + returns[1:]
        var_2 = np.var(returns_2)
        
        if var_1 > 0:
            return float(var_2 / (2 * var_1))
        return 1.0

class WaveletAnalyzer:
    """Analyse par ondelettes"""
    
    def __init__(self):
        self.scales = np.arange(1, 50)
    
    def analyze(self, prices: List[float]) -> Dict:
        if len(prices) < 50:
            return {}
        
        prices_np = np.array(prices)
        
        # Transformée en ondelettes simplifiée
        wavelet_coeffs = []
        for scale in self.scales:
            if len(prices_np) > scale * 2:
                # Filtre passe-bande simple
                filtered = savgol_filter(prices_np, scale * 2 + 1, 3)
                coeff = np.std(filtered)
                wavelet_coeffs.append(coeff)
            else:
                wavelet_coeffs.append(0)
        
        wavelet_coeffs = np.array(wavelet_coeffs)
        
        return {
            'scales': self.scales.tolist(),
            'coefficients': wavelet_coeffs.tolist(),
            'dominant_scale': int(self.scales[np.argmax(wavelet_coeffs)]) if len(wavelet_coeffs) > 0 else 1,
            'scale_energy': float(np.sum(wavelet_coeffs**2)),
            'scale_entropy': float(entropy(wavelet_coeffs + 1e-10)),
            'multiscale_correlation': self.calculate_multiscale_correlation(wavelet_coeffs)
        }
    
    def calculate_multiscale_correlation(self, coeffs: np.ndarray) -> float:
        """Corrélation entre différentes échelles"""
        if len(coeffs) < 3:
            return 0.0
        
        half = len(coeffs) // 2
        low_scale = coeffs[:half]
        high_scale = coeffs[half:]
        
        if len(low_scale) > 1 and len(high_scale) > 1:
            return float(np.corrcoef(low_scale, high_scale[:len(low_scale)])[0, 1])
        return 0.0

class HilbertAnalyzer:
    """Analyse de Hilbert-Huang"""
    
    def analyze(self, prices: List[float]) -> Dict:
        if len(prices) < 20:
            return {}
        
        prices_np = np.array(prices)
        
        try:
            analytic_signal = hilbert(prices_np)
            amplitude = np.abs(analytic_signal)
            phase = np.unwrap(np.angle(analytic_signal))
            frequency = np.diff(phase) / (2 * np.pi)
            
            return {
                'instantaneous_amplitude_mean': float(np.mean(amplitude)),
                'instantaneous_amplitude_std': float(np.std(amplitude)),
                'instantaneous_frequency_mean': float(np.mean(frequency)) if len(frequency) > 0 else 0,
                'instantaneous_frequency_std': float(np.std(frequency)) if len(frequency) > 0 else 0,
                'coherence': self.calculate_coherence(amplitude, frequency),
                'phase_synchronization': self.calculate_phase_sync(phase)
            }
        except:
            return {}
    
    def calculate_coherence(self, amplitude: np.ndarray, frequency: np.ndarray) -> float:
        if len(amplitude) < 10 or len(frequency) < 10:
            return 0.0
        
        try:
            # Cohérence amplitude-fréquence
            corr = np.corrcoef(amplitude[-10:], frequency[-10:])[0, 1]
            return float(abs(corr))
        except:
            return 0.0
    
    def calculate_phase_sync(self, phase: np.ndarray) -> float:
        if len(phase) < 10:
            return 0.0
        
        # Synchronisation de phase
        phase_diff = np.diff(phase)
        phase_lock = np.mean(np.cos(phase_diff))
        return float(abs(phase_lock))

class FractalAnalyzer:
    """Analyse fractale"""
    
    def analyze(self, prices: List[float]) -> Dict:
        if len(prices) < 50:
            return {}
        
        prices_np = np.array(prices)
        
        return {
            'fractal_dimension': self.calculate_fractal_dimension(prices_np),
            'hurst_exponent': self.calculate_hurst(prices_np),
            'multifractal_spectrum': self.calculate_multifractal(prices_np),
            'lacunarity': self.calculate_lacunarity(prices_np)
        }
    
    def calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """Dimension fractale par box-counting"""
        n = len(data)
        if n < 10:
            return 1.0
        
        scales = np.logspace(np.log10(2), np.log10(n//2), 10, dtype=int)
        scales = scales[scales > 1]
        
        counts = []
        for scale in scales:
            boxes = np.array_split(data, scale)
            box_count = sum(1 for box in boxes if len(box) > 0 and np.ptp(box) > 0)
            counts.append(box_count)
        
        if len(counts) < 2:
            return 1.0
        
        try:
            log_scales = np.log(scales[:len(counts)])
            log_counts = np.log(counts)
            coeff = np.polyfit(log_scales, log_counts, 1)
            return float(coeff[0])
        except:
            return 1.0
    
    def calculate_hurst(self, data: np.ndarray) -> float:
        """Exposant de Hurst R/S"""
        n = len(data)
        if n < 100:
            return 0.5
        
        rs_values = []
        scales = []
        
        for scale in range(10, n//2, 10):
            segments = n // scale
            if segments < 2:
                continue
            
            rs_segment = []
            for i in range(segments):
                segment = data[i*scale:(i+1)*scale]
                mean = np.mean(segment)
                deviation = segment - mean
                z = np.cumsum(deviation)
                r = np.max(z) - np.min(z)
                s = np.std(segment)
                
                if s > 0:
                    rs_segment.append(r / s)
            
            if rs_segment:
                rs_values.append(np.mean(rs_segment))
                scales.append(scale)
        
        if len(scales) < 2:
            return 0.5
        
        try:
            hurst = np.polyfit(np.log(scales), np.log(rs_values), 1)[0]
            return float(hurst)
        except:
            return 0.5
    
    def calculate_multifractal(self, data: np.ndarray) -> Dict:
        """Spectre multifractal simplifié"""
        n = len(data)
        if n < 100:
            return {'width': 0.0, 'asymmetry': 0.0}
        
        q_values = [-5, -2, -1, 0, 1, 2, 5]
        tau_q = []
        
        for q in q_values:
            if q == 0:
                # Cas spécial q=0
                tau_q.append(-self.calculate_fractal_dimension(data))
            else:
                # Approximation
                hurst = self.calculate_hurst(data)
                tau_q.append(q * hurst - 1)
        
        # Largeur du spectre
        width = max(tau_q) - min(tau_q)
        
        # Asymétrie
        left = abs(tau_q[0] - tau_q[3])  # q=-5 à q=0
        right = abs(tau_q[-1] - tau_q[3])  # q=5 à q=0
        asymmetry = (right - left) / (right + left + 1e-10)
        
        return {
            'width': float(width),
            'asymmetry': float(asymmetry),
            'tau_q': [float(t) for t in tau_q]
        }
    
    def calculate_lacunarity(self, data: np.ndarray) -> float:
        """Lacunarité - mesure des lacunes"""
        n = len(data)
        if n < 50:
            return 0.0
        
        # Binarisation
        threshold = np.median(data)
        binary = (data > threshold).astype(int)
        
        # Calcul de lacunarité sur différentes fenêtres
        lacunarities = []
        for window in [3, 5, 7, 10]:
            if n >= window:
                means = []
                for i in range(n - window + 1):
                    means.append(np.mean(binary[i:i+window]))
                
                if len(means) > 0:
                    mean_val = np.mean(means)
                    std_val = np.std(means)
                    if mean_val > 0:
                        lacunarities.append((std_val / mean_val) ** 2)
        
        return float(np.mean(lacunarities)) if lacunarities else 0.0

class EntropyAnalyzer:
    """Analyse d'entropie et de complexité"""
    
    def analyze(self, prices: List[float]) -> Dict:
        if len(prices) < 20:
            return {}
        
        prices_np = np.array(prices)
        returns = np.diff(prices_np) / prices_np[:-1]
        
        return {
            'shannon_entropy': self.shannon_entropy(returns),
            'approximate_entropy': self.approximate_entropy(prices_np),
            'sample_entropy': self.sample_entropy(prices_np),
            'permutation_entropy': self.permutation_entropy(prices_np),
            'multiscale_entropy': self.multiscale_entropy(prices_np),
            'complexity': self.complexity_measure(prices_np)
        }
    
    def shannon_entropy(self, data: np.ndarray, bins: int = 20) -> float:
        """Entropie de Shannon classique"""
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log(hist)))
    
    def approximate_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Entropie approximative (Pincus)"""
        n = len(data)
        if n < m + 1:
            return 0.0
        
        def _phi(m_val):
            patterns = []
            for i in range(n - m_val + 1):
                patterns.append(data[i:i+m_val])
            
            patterns = np.array(patterns)
            c = []
            for i in range(len(patterns)):
                dist = np.max(np.abs(patterns - patterns[i]), axis=1)
                c.append(np.sum(dist <= r * np.std(data)) / (n - m_val + 1))
            
            return np.sum(np.log(c)) / (n - m_val + 1)
        
        return float(_phi(m) - _phi(m + 1))
    
    def sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Entropie échantillon (amélioration de ApEn)"""
        n = len(data)
        if n < m + 1:
            return 0.0
        
        def _count_matches(m_val):
            patterns = []
            for i in range(n - m_val):
                patterns.append(data[i:i+m_val])
            
            patterns = np.array(patterns)
            count = 0
            for i in range(len(patterns)):
                for j in range(i+1, len(patterns)):
                    if np.max(np.abs(patterns[i] - patterns[j])) <= r * np.std(data):
                        count += 1
            
            return count
        
        A = _count_matches(m + 1)
        B = _count_matches(m)
        
        if B > 0 and A > 0:
            return float(-np.log(A / B))
        return 0.0
    
    def permutation_entropy(self, data: np.ndarray, n: int = 3) -> float:
        """Entropie de permutation"""
        if len(data) < n:
            return 0.0
        
        permutations = []
        for i in range(len(data) - n + 1):
            segment = data[i:i+n]
            permutations.append(tuple(np.argsort(segment)))
        
        unique, counts = np.unique(permutations, return_counts=True)
        probs = counts / len(permutations)
        probs = probs[probs > 0]
        
        return float(-np.sum(probs * np.log(probs)))
    
    def multiscale_entropy(self, data: np.ndarray, max_scale: int = 5) -> List[float]:
        """Entropie multi-échelle"""
        entropies = []
        for scale in range(1, max_scale + 1):
            coarse = self.coarse_grain(data, scale)
            if len(coarse) > 10:
                entropies.append(self.sample_entropy(coarse))
        
        return [float(e) for e in entropies]
    
    def coarse_grain(self, data: np.ndarray, scale: int) -> np.ndarray:
        """Grain grossier pour l'entropie multi-échelle"""
        n = len(data)
        coarse = np.zeros(n // scale)
        for i in range(len(coarse)):
            coarse[i] = np.mean(data[i*scale:(i+1)*scale])
        return coarse
    
    def complexity_measure(self, data: np.ndarray) -> float:
        """Mesure de complexité Lempel-Ziv simplifiée"""
        if len(data) < 10:
            return 0.0
        
        # Binarisation
        threshold = np.median(data)
        binary = ''.join(['1' if x > threshold else '0' for x in data])
        
        # Complexité Lempel-Ziv
        complexity = 0
        n = len(binary)
        substrings = set()
        
        i = 0
        while i < n:
            j = i + 1
            while j <= n and binary[i:j] in substrings:
                j += 1
            
            if j <= n:
                substrings.add(binary[i:j])
                complexity += 1
            
            i = j
        
        # Normalisation
        max_complexity = n / np.log2(n) if n > 1 else 1
        return float(complexity / max_complexity)

# ============================================================================
# CLASSIFICATEURS SOCIAUX
# ============================================================================

class SocialClassifier:
    """Classificateur social des bougies"""
    
    def __init__(self, classes: List[str]):
        self.classes = classes
        self.scaler = StandardScaler()
        self.classifier = KMeans(n_clusters=len(classes), random_state=42)
        self.is_trained = False
        
    def classify(self, prices: List[float], volumes: List[float]) -> Dict:
        """Classifie la bougie actuelle"""
        
        if len(prices) < 20 or len(volumes) < 20:
            return {'class': 'UNKNOWN', 'confidence': 0.0, 'features': {}}
        
        # Extraction de features
        features = self.extract_features(prices, volumes)
        
        if not self.is_trained:
            # Classification heuristique initiale
            return self.heuristic_classification(features)
        else:
            # Classification par ML
            return self.ml_classification(features)
    
    def extract_features(self, prices: List[float], volumes: List[float]) -> Dict:
        """Extrait les features de classification"""
        
        prices_np = np.array(prices)
        volumes_np = np.array(volumes)
        
        # Features de prix
        returns = np.diff(prices_np) / prices_np[:-1]
        
        features = {
            'price_mean': float(np.mean(prices_np)),
            'price_std': float(np.std(prices_np)),
            'price_skew': float(skew(prices_np)) if len(prices_np) > 2 else 0,
            'price_kurtosis': float(kurtosis(prices_np)) if len(prices_np) > 3 else 0,
            
            'return_mean': float(np.mean(returns)) if len(returns) > 0 else 0,
            'return_std': float(np.std(returns)) if len(returns) > 0 else 0,
            'return_skew': float(skew(returns)) if len(returns) > 2 else 0,
            'return_kurtosis': float(kurtosis(returns)) if len(returns) > 3 else 0,
            
            'volume_mean': float(np.mean(volumes_np)),
            'volume_std': float(np.std(volumes_np)),
            'volume_skew': float(skew(volumes_np)) if len(volumes_np) > 2 else 0,
            'volume_kurtosis': float(kurtosis(volumes_np)) if len(volumes_np) > 3 else 0,
            
            'price_volume_corr': float(np.corrcoef(prices_np[-len(volumes_np):], volumes_np)[0, 1]) 
                if len(prices_np) == len(volumes_np) else 0,
            
            'trend_strength': self.calculate_trend_strength(prices_np),
            'volatility': float(np.std(returns) * 100) if len(returns) > 0 else 0,
            'mobility': float((prices_np[-1] - prices_np[0]) * 10000) if len(prices_np) > 1 else 0,
            'aggression': self.calculate_aggression(prices_np, volumes_np)
        }
        
        return features
    
    def calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calcule la force de la tendance"""
        if len(prices) < 10:
            return 0.0
        
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = linregress(x, prices)
        
        # Normalisation
        price_range = np.ptp(prices)
        if price_range > 0:
            normalized_slope = abs(slope) * len(prices) / price_range
            return float(normalized_slope * r_value**2)
        
        return 0.0
    
    def calculate_aggression(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calcule le niveau d'agression"""
        if len(prices) < 5 or len(volumes) < 5:
            return 0.0
        
        # Agression = volatilité * volume relatif
        returns = np.diff(prices[-5:]) / prices[-6:-1]
        volatility = np.std(returns) * 100
        
        volume_mean = np.mean(volumes[-5:])
        volume_std = np.std(volumes[-5:])
        
        if volume_mean > 0:
            volume_ratio = volume_std / volume_mean
        else:
            volume_ratio = 0
        
        return float(volatility * volume_ratio)
    
    def heuristic_classification(self, features: Dict) -> Dict:
        """Classification heuristique basée sur les features"""
        
        # Règles de classification
        mobility = features.get('mobility', 0)
        volatility = features.get('volatility', 0)
        trend_strength = features.get('trend_strength', 0)
        aggression = features.get('aggression', 0)
        
        # Classification
        if mobility < -80 and aggression > 30:
            social_class = "PAUVRE_REVOLTE"
            confidence = 0.8
        
        elif mobility < -30 and volatility > 20:
            social_class = "PAUVRE_SOUMIS"
            confidence = 0.7
        
        elif abs(mobility) < 20 and trend_strength < 0.3:
            social_class = "CLASSE_MOYENNE_STABLE"
            confidence = 0.6
        
        elif abs(mobility) < 40 and volatility > 15:
            social_class = "CLASSE_MOYENNE_ANXIEUSE"
            confidence = 0.65
        
        elif mobility > 60 and volatility < 10:
            social_class = "RICHE_CALME"
            confidence = 0.75
        
        elif mobility > 40 and aggression > 25:
            social_class = "RICHE_AGITE"
            confidence = 0.7
        
        elif volatility > 25 and trend_strength < 0.2:
            social_class = "VAMPIRE_SANG_FROID"
            confidence = 0.6
        
        elif volatility > 40 and aggression > 40:
            social_class = "VAMPIRE_AFFAME"
            confidence = 0.65
        
        elif mobility > 100 and volatility < 15:
            social_class = "TRAITRE_CALCULATEUR"
            confidence = 0.7
        
        elif mobility < -100 and aggression > 50:
            social_class = "TRAITRE_PANIQUE"
            confidence = 0.65
        
        else:
            social_class = "CLASSE_MOYENNE_STABLE"
            confidence = 0.5
        
        return {
            'class': social_class,
            'confidence': confidence,
            'features': features,
            'method': 'heuristic'
        }
    
    def ml_classification(self, features: Dict) -> Dict:
        """Classification par machine learning"""
        # À implémenter avec entraînement
        return self.heuristic_classification(features)
    
    def train(self, training_data: List[Dict]):
        """Entraîne le classificateur"""
        # À implémenter
        pass

class PersonalityClassifier:
    """Classificateur de personnalité MBTI"""
    
    def classify(self, candle_data: Dict) -> Dict:
        """Classe la personnalité de la bougie"""
        
        # Features de personnalité
        volatility = candle_data.get('volatility', 0)
        predictability = candle_data.get('predictability', 0)
        aggression = candle_data.get('aggression', 0)
        social_class = candle_data.get('social_class', '')
        
        # Dimensions MBTI
        # E/I: Extraversion/Introversion - Volatilité
        # S/N: Sensing/Intuition - Prévisibilité
        # T/F: Thinking/Feeling - Agression
        # J/P: Judging/Perceiving - Stabilité
        
        ei = 'E' if volatility > 0.5 else 'I'
        sn = 'S' if predictability > 0.6 else 'N'
        tf = 'T' if aggression > 0.4 else 'F'
        jp = 'J' if 'STABLE' in social_class or 'CALME' in social_class else 'P'
        
        mbti_type = f"{ei}{sn}{tf}{jp}"
        
        # Description
        descriptions = {
            'INTJ': 'Architecte - Stratège calculateur',
            'ENTJ': 'Commandant - Leader agressif',
            'INTP': 'Logicien - Analyste froid',
            'ENTP': 'Innovateur - Opportuniste',
            'INFJ': 'Conseiller - Sage patient',
            'ENFJ': 'Mentor - Guide charismatique',
            'INFP': 'Médiateur - Idéaliste',
            'ENFP': 'Inspirateur - Enthousiaste',
            'ISTJ': 'Logisticien - Méthodique',
            'ESTJ': 'Directeur - Organisateur',
            'ISFJ': 'Défenseur - Protecteur',
            'ESFJ': 'Consul - Nourricier',
            'ISTP': 'Virtuose - Pragmatique',
            'ESTP': 'Entrepreneur - Audacieux',
            'ISFP': 'Aventurier - Artistique',
            'ESFP': 'Amuseur - Spontané'
        }
        
        return {
            'mbti_type': mbti_type,
            'description': descriptions.get(mbti_type, 'Type inconnu'),
            'dimensions': {
                'EI': ei,
                'SN': sn,
                'TF': tf,
                'JP': jp
            },
            'scores': {
                'volatility': volatility,
                'predictability': predictability,
                'aggression': aggression
            }
        }

class MarketRegimeClassifier:
    """Classificateur de régime de marché"""
    
    def classify(self, analysis_results: Dict) -> Dict:
        """Classe le régime de marché actuel"""
        
        # Extraction des métriques
        quantum_state = analysis_results.get('quantum_state', '')
        volatility = analysis_results.get('volatility', 0)
        trend_strength = analysis_results.get('trend_strength', 0)
        coherence = analysis_results.get('coherence', 0)
        entropy = analysis_results.get('entropy', 0)
        
        # Classification
        if trend_strength > 0.7 and coherence > 0.6:
            regime = "TREND_FORT"
            confidence = 0.8
        
        elif trend_strength < 0.3 and volatility < 0.2:
            regime = "RANGE_SERRÉ"
            confidence = 0.7
        
        elif volatility > 0.6 and entropy > 0.7:
            regime = "CHAOS_ÉLEVÉ"
            confidence = 0.75
        
        elif volatility > 0.4 and trend_strength < 0.4:
            regime = "VOLATILITÉ_ÉLEVÉE"
            confidence = 0.65
        
        elif 'REVOLUTION' in quantum_state:
            regime = "RÉVOLUTION_MARCHÉ"
            confidence = 0.7
        
        elif 'COHERENCE' in quantum_state:
            regime = "COHÉRENCE_QUANTIQUE"
            confidence = 0.75
        
        elif 'TUNNELING' in quantum_state:
            regime = "EFFET_TUNNEL"
            confidence = 0.6
        
        else:
            regime = "RANGE_NORMAL"
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': confidence,
            'characteristics': {
                'volatility_level': 'HIGH' if volatility > 0.4 else 'LOW',
                'trending': 'YES' if trend_strength > 0.5 else 'NO',
                'chaotic': 'YES' if entropy > 0.6 else 'NO',
                'coherent': 'YES' if coherence > 0.5 else 'NO'
            },
            'trading_implications': self.get_trading_implications(regime)
        }
    
    def get_trading_implications(self, regime: str) -> List[str]:
        """Retourne les implications trading pour le régime"""
        
        implications = {
            "TREND_FORT": [
                "Suivre la tendance principale",
                "Éviter les contre-tendances",
                "Utiliser des trailing stops agressifs",
                "Allonger la durée des trades"
            ],
            "RANGE_SERRÉ": [
                "Scalping sur les supports/résistances",
                "Petits take profits",
                "Trading mean reversion",
                "Éviter les breakouts prématurés"
            ],
            "CHAOS_ÉLEVÉ": [
                "Réduire la taille des positions",
                "Augmenter les stops",
                "Éviter les trades directionnels",
                "Attendre la clarification"
            ],
            "VOLATILITÉ_ÉLEVÉE": [
                "Profiter des grands mouvements",
                "Utiliser des stops larges",
                "Trading momentum",
                "Être prêt à inverser rapidement"
            ],
            "RÉVOLUTION_MARCHÉ": [
                "Être très agressif",
                "Suivre la nouvelle direction",
                "Ignorer les anciens niveaux",
                "Capitaliser sur la panique"
            ],
            "COHÉRENCE_QUANTIQUE": [
                "Confiance élevée dans les signaux",
                "Prendre plus de trades",
                "Réduire les stops",
                "Profiter de la prévisibilité"
            ],
            "EFFET_TUNNEL": [
                "Attendre la percée",
                "Positionner des ordres limites",
                "Être patient",
                "Préparer le momentum post-percée"
            ],
            "RANGE_NORMAL": [
                "Trading range classique",
                "Vendre résistance, acheter support",
                "Gestion de risque standard",
                "Approche équilibrée"
            ]
        }
        
        return implications.get(regime, ["Approche standard"])

# ============================================================================
# DÉTECTEURS ET PRÉDICTEURS
# ============================================================================

class AnomalyDetector:
    """Détecteur d'anomalies et d'opportunités"""
    
    def detect(self, prices: List[float], volumes: List[float] = None) -> Dict:
        """Détecte les anomalies dans les prix"""
        
        if len(prices) < 50:
            return {'anomalies': [], 'opportunities': []}
        
        prices_np = np.array(prices)
        
        anomalies = []
        opportunities = []
        
        # 1. Détection de spikes
        spikes = self.detect_spikes(prices_np)
        anomalies.extend(spikes)
        
        # 2. Détection de changements de régime
        regime_changes = self.detect_regime_changes(prices_np)
        anomalies.extend(regime_changes)
        
        # 3. Détection de divergences
        divergences = self.detect_divergences(prices_np)
        opportunities.extend(divergences)
        
        # 4. Détection de clusters d'activité
        clusters = self.detect_activity_clusters(prices_np, volumes)
        opportunities.extend(clusters)
        
        # 5. Détection de niveaux critiques
        critical_levels = self.detect_critical_levels(prices_np)
        opportunities.extend(critical_levels)
        
        return {
            'anomalies': anomalies,
            'opportunities': opportunities,
            'anomaly_count': len(anomalies),
            'opportunity_count': len(opportunities),
            'anomaly_score': self.calculate_anomaly_score(anomalies),
            'opportunity_score': self.calculate_opportunity_score(opportunities)
        }
    
    def detect_spikes(self, prices: np.ndarray, threshold: float = 3.0) -> List[Dict]:
        """Détecte les spikes de prix"""
        spikes = []
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) > 10:
            mean_return = np.mean(np.abs(returns[-20:])) if len(returns) >= 20 else np.mean(np.abs(returns))
            std_return = np.std(np.abs(returns[-20:])) if len(returns) >= 20 else np.std(np.abs(returns))
            
            for i in range(len(returns)):
                if abs(returns[i]) > mean_return + threshold * std_return:
                    spikes.append({
                        'type': 'SPIKE',
                        'direction': 'UP' if returns[i] > 0 else 'DOWN',
                        'magnitude': float(abs(returns[i])),
                        'position': i,
                        'timestamp': datetime.now().timestamp() - (len(returns) - i) * 0.001
                    })
        
        return spikes[-5:]  # Retourne les 5 derniers spikes
    
    def detect_regime_changes(self, prices: np.ndarray, window: int = 20) -> List[Dict]:
        """Détecte les changements de régime"""
        changes = []
        
        if len(prices) > window * 2:
            for i in range(window, len(prices) - window):
                before = prices[i-window:i]
                after = prices[i:i+window]
                
                # Statistiques avant/après
                mean_before = np.mean(before)
                mean_after = np.mean(after)
                std_before = np.std(before)
                std_after = np.std(after)
                
                # Test de changement
                mean_change = abs(mean_after - mean_before) / (std_before + 1e-10)
                vol_change = abs(std_after - std_before) / (std_before + 1e-10)
                
                if mean_change > 2.0 or vol_change > 1.5:
                    changes.append({
                        'type': 'REGIME_CHANGE',
                        'at_position': i,
                        'mean_change': float(mean_change),
                        'vol_change': float(vol_change),
                        'new_regime': 'HIGHER_VOL' if vol_change > 1.5 else 'TREND_CHANGE'
                    })
        
        return changes[-3:]  # Retourne les 3 derniers changements
    
    def detect_divergences(self, prices: np.ndarray) -> List[Dict]:
        """Détecte les divergences prix/indicateurs"""
        divergences = []
        
        if len(prices) > 30:
            # RSI
            rsi = self.calculate_rsi(prices, period=14)
            
            # Détection de divergence RSI
            for lookback in [10, 15, 20]:
                if len(prices) > lookback * 2:
                    price_trend = prices[-lookback] - prices[-lookback*2]
                    rsi_trend = rsi[-lookback] - rsi[-lookback*2]
                    
                    if price_trend * rsi_trend < 0 and abs(price_trend) > 0.001 and abs(rsi_trend) > 5:
                        divergences.append({
                            'type': 'DIVERGENCE',
                            'indicator': 'RSI',
                            'price_trend': float(price_trend),
                            'indicator_trend': float(rsi_trend),
                            'lookback': lookback,
                            'signal': 'BEARISH' if price_trend > 0 and rsi_trend < 0 else 'BULLISH'
                        })
        
        return divergences
    
    def detect_activity_clusters(self, prices: np.ndarray, volumes: np.ndarray = None) -> List[Dict]:
        """Détecte les clusters d'activité"""
        clusters = []
        
        if len(prices) > 50:
            # Volume clustering (si disponible)
            if volumes is not None and len(volumes) > 50:
                volume_mean = np.mean(volumes[-50:])
                volume_std = np.std(volumes[-50:])
                
                high_volume_indices = np.where(volumes[-50:] > volume_mean + volume_std)[0]
                
                if len(high_volume_indices) > 0:
                    clusters.append({
                        'type': 'VOLUME_CLUSTER',
                        'count': len(high_volume_indices),
                        'intensity': float(np.mean(volumes[-50:][high_volume_indices]) / volume_mean),
                        'positions': [int(i) for i in high_volume_indices]
                    })
            
            # Price clustering
            hist, bin_edges = np.histogram(prices[-50:], bins=10)
            cluster_bins = np.where(hist > np.mean(hist) * 1.5)[0]
            
            if len(cluster_bins) > 0:
                cluster_levels = []
                for bin_idx in cluster_bins:
                    level = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
                    strength = hist[bin_idx] / len(prices[-50:])
                    cluster_levels.append({
                        'level': float(level),
                        'strength': float(strength)
                    })
                
                clusters.append({
                    'type': 'PRICE_CLUSTER',
                    'levels': cluster_levels,
                    'dominant_level': cluster_levels[0] if cluster_levels else None
                })
        
        return clusters
    
    def detect_critical_levels(self, prices: np.ndarray) -> List[Dict]:
        """Détecte les niveaux de prix critiques"""
        levels = []
        
        if len(prices) > 100:
            # Supports et résistances
            window = 20
            highs = []
            lows = []
            
            for i in range(window, len(prices) - window):
                if prices[i] == np.max(prices[i-window:i+window]):
                    highs.append({
                        'level': float(prices[i]),
                        'position': i,
                        'strength': self.calculate_level_strength(prices, prices[i], window)
                    })
                elif prices[i] == np.min(prices[i-window:i+window]):
                    lows.append({
                        'level': float(prices[i]),
                        'position': i,
                        'strength': self.calculate_level_strength(prices, prices[i], window)
                    })
            
            # Garder les niveaux les plus forts
            if highs:
                strongest_high = max(highs, key=lambda x: x['strength'])
                levels.append({
                    'type': 'RESISTANCE',
                    'level': strongest_high['level'],
                    'strength': strongest_high['strength'],
                    'distance': float(abs(prices[-1] - strongest_high['level']))
                })
            
            if lows:
                strongest_low = max(lows, key=lambda x: x['strength'])
                levels.append({
                    'type': 'SUPPORT',
                    'level': strongest_low['level'],
                    'strength': strongest_low['strength'],
                    'distance': float(abs(prices[-1] - strongest_low['level']))
                })
            
            # Niveaux psychologiques
            current_price = prices[-1]
            psychological_levels = [round(current_price, 0), round(current_price, 1), round(current_price, 2)]
            
            for level in psychological_levels:
                levels.append({
                    'type': 'PSYCHOLOGICAL',
                    'level': float(level),
                    'distance': float(abs(current_price - level))
                })
        
        return levels
    
    def calculate_level_strength(self, prices: np.ndarray, level: float, window: int) -> float:
        """Calcule la force d'un niveau de support/résistance"""
        touches = 0
        for i in range(len(prices) - window, len(prices)):
            if abs(prices[i] - level) < 0.0005:  # 0.5 pips
                touches += 1
        
        return float(touches / window)
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calcule le RSI"""
        if len(prices) < period + 1:
            return np.zeros_like(prices)
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    def calculate_anomaly_score(self, anomalies: List[Dict]) -> float:
        """Calcule un score d'anomalie global"""
        if not anomalies:
            return 0.0
        
        score = 0.0
        weights = {
            'SPIKE': 0.4,
            'REGIME_CHANGE': 0.3,
            'DIVERGENCE': 0.2,
            'VOLUME_CLUSTER': 0.1,
            'PRICE_CLUSTER': 0.1
        }
        
        for anomaly in anomalies:
            anomaly_type = anomaly.get('type', '')
            weight = weights.get(anomaly_type, 0.05)
            
            if anomaly_type == 'SPIKE':
                magnitude = anomaly.get('magnitude', 0)
                score += weight * min(magnitude * 1000, 1.0)
            
            elif anomaly_type == 'REGIME_CHANGE':
                mean_change = anomaly.get('mean_change', 0)
                score += weight * min(mean_change / 3.0, 1.0)
        
        return min(score, 1.0)
    
    def calculate_opportunity_score(self, opportunities: List[Dict]) -> float:
        """Calcule un score d'opportunité global"""
        if not opportunities:
            return 0.0
        
        score = 0.0
        weights = {
            'DIVERGENCE': 0.3,
            'VOLUME_CLUSTER': 0.2,
            'PRICE_CLUSTER': 0.2,
            'RESISTANCE': 0.15,
            'SUPPORT': 0.15,
            'PSYCHOLOGICAL': 0.1
        }
        
        for opp in opportunities:
            opp_type = opp.get('type', '')
            weight = weights.get(opp_type, 0.05)
            
            if opp_type in ['RESISTANCE', 'SUPPORT', 'PSYCHOLOGICAL']:
                strength = opp.get('strength', 0.5)
                distance = opp.get('distance', 0)
                
                # Plus proche = meilleur
                distance_score = max(0, 1 - (distance * 10000) / 50)  # 50 pips max
                score += weight * strength * distance_score
            
            elif opp_type == 'DIVERGENCE':
                signal_strength = abs(opp.get('price_trend', 0)) * 10000
                score += weight * min(signal_strength / 20, 1.0)  # 20 pips
        
        return min(score, 1.0)

class PatternDetector:
    """Détecteur de patterns graphiques"""
    
    def __init__(self):
        self.patterns = {
            'ENGULFING': self.detect_engulfing,
            'HARAMI': self.detect_harami,
            'HAMMER': self.detect_hammer,
            'SHOOTING_STAR': self.detect_shooting_star,
            'DOJI': self.detect_doji,
            'THREE_WHITE_SOLDIERS': self.detect_white_soldiers,
            'THREE_BLACK_CROWS': self.detect_black_crows,
            'MORNING_STAR': self.detect_morning_star,
            'EVENING_STAR': self.detect_evening_star,
            'HEAD_SHOULDERS': self.detect_head_shoulders,
            'INVERSE_HEAD_SHOULDERS': self.detect_inverse_head_shoulders,
            'DOUBLE_TOP': self.detect_double_top,
            'DOUBLE_BOTTOM': self.detect_double_bottom,
            'TRIANGLE': self.detect_triangle,
            'WEDGE': self.detect_wedge,
            'FLAG': self.detect_flag,
            'PENNANT': self.detect_pennant
        }
    
    def detect(self, candles: List[Dict]) -> Dict:
        """Détecte tous les patterns dans les bougies"""
        
        if len(candles) < 3:
            return {'patterns': [], 'strong_patterns': []}
        
        detected_patterns = []
        
        # Détecter chaque type de pattern
        for pattern_name, detector_func in self.patterns.items():
            patterns = detector_func(candles)
            detected_patterns.extend(patterns)
        
        # Filtrer les patterns forts
        strong_patterns = [p for p in detected_patterns if p.get('strength', 0) > 0.7]
        
        return {
            'patterns': detected_patterns,
            'strong_patterns': strong_patterns,
            'pattern_count': len(detected_patterns),
            'strong_pattern_count': len(strong_patterns),
            'pattern_score': self.calculate_pattern_score(detected_patterns)
        }
    
    def detect_engulfing(self, candles: List[Dict], lookback: int = 20) -> List[Dict]:
        """Détecte les patterns engulfing"""
        patterns = []
        
        if len(candles) >= 2:
            for i in range(1, min(len(candles), lookback)):
                prev = candles[i-1]
                curr = candles[i]
                
                # Bullish engulfing
                if (prev['close'] < prev['open'] and  # Bougie baissière précédente
                    curr['close'] > curr['open'] and  # Bougie haussière actuelle
                    curr['open'] < prev['close'] and  # Ouverture sous la clôture précédente
                    curr['close'] > prev['open']):    # Clôture au-dessus de l'ouverture précédente
                    
                    body_ratio = abs(curr['close'] - curr['open']) / abs(prev['close'] - prev['open'])
                    strength = min(body_ratio, 2.0) / 2.0  # Normalisé à 0-1
                    
                    patterns.append({
                        'type': 'BULLISH_ENGULFING',
                        'position': i,
                        'strength': strength,
                        'confidence': 0.6 * strength,
                        'price_action': 'REVERSAL'
                    })
                
                # Bearish engulfing
                elif (prev['close'] > prev['open'] and  # Bougie haussière précédente
                      curr['close'] < curr['open'] and  # Bougie baissière actuelle
                      curr['open'] > prev['close'] and  # Ouverture au-dessus de la clôture précédente
                      curr['close'] < prev['open']):    # Clôture sous l'ouverture précédente
                    
                    body_ratio = abs(curr['close'] - curr['open']) / abs(prev['close'] - prev['open'])
                    strength = min(body_ratio, 2.0) / 2.0
                    
                    patterns.append({
                        'type': 'BEARISH_ENGULFING',
                        'position': i,
                        'strength': strength,
                        'confidence': 0.6 * strength,
                        'price_action': 'REVERSAL'
                    })
        
        return patterns
    
    def detect_harami(self, candles: List[Dict], lookback: int = 20) -> List[Dict]:
        """Détecte les patterns harami"""
        patterns = []
        
        if len(candles) >= 2:
            for i in range(1, min(len(candles), lookback)):
                prev = candles[i-1]
                curr = candles[i]
                
                # Bullish harami
                if (prev['close'] < prev['open'] and  # Bougie baissière précédente (grande)
                    curr['close'] > curr['open'] and  # Bougie haussière actuelle (petite)
                    curr['open'] > prev['close'] and  # Ouverture au-dessus de la clôture précédente
                    curr['close'] < prev['open']):    # Clôture sous l'ouverture précédente
                    
                    patterns.append({
                        'type': 'BULLISH_HARAMI',
                        'position': i,
                        'strength': 0.6,
                        'confidence': 0.5,
                        'price_action': 'REVERSAL'
                    })
                
                # Bearish harami
                elif (prev['close'] > prev['open'] and  # Bougie haussière précédente (grande)
                      curr['close'] < curr['open'] and  # Bougie baissière actuelle (petite)
                      curr['open'] < prev['close'] and  # Ouverture sous la clôture précédente
                      curr['close'] > prev['open']):    # Clôture au-dessus de l'ouverture précédente
                    
                    patterns.append({
                        'type': 'BEARISH_HARAMI',
                        'position': i,
                        'strength': 0.6,
                        'confidence': 0.5,
                        'price_action': 'REVERSAL'
                    })
        
        return patterns
    
    def detect_hammer(self, candles: List[Dict], lookback: int = 20) -> List[Dict]:
        """Détecte les marteaux et hanging man"""
        patterns = []
        
        if len(candles) >= 1:
            for i in range(min(len(candles), lookback)):
                candle = candles[i]
                
                body = abs(candle['close'] - candle['open'])
                total_range = candle['high'] - candle['low']
                
                if total_range > 0:
                    body_ratio = body / total_range
                    lower_wick = (min(candle['open'], candle['close']) - candle['low']) / total_range
                    upper_wick = (candle['high'] - max(candle['open'], candle['close'])) / total_range
                    
                    # Hammer (marteau)
                    if lower_wick > 0.6 and body_ratio < 0.3 and upper_wick < 0.1:
                        patterns.append({
                            'type': 'HAMMER',
                            'position': i,
                            'strength': lower_wick,
                            'confidence': 0.7,
                            'price_action': 'BULLISH_REVERSAL'
                        })
                    
                    # Inverted hammer (marteau inversé)
                    elif upper_wick > 0.6 and body_ratio < 0.3 and lower_wick < 0.1:
                        patterns.append({
                            'type': 'INVERTED_HAMMER',
                            'position': i,
                            'strength': upper_wick,
                            'confidence': 0.6,
                            'price_action': 'BULLISH_REVERSAL'
                        })
                    
                    # Hanging man (pendu)
                    elif lower_wick > 0.6 and body_ratio < 0.3 and upper_wick < 0.1:
                        # Contexte de tendance haussière
                        if i > 0 and candles[i-1]['close'] < candles[i-1]['open']:
                            patterns.append({
                                'type': 'HANGING_MAN',
                                'position': i,
                                'strength': lower_wick,
                                'confidence': 0.7,
                                'price_action': 'BEARISH_REVERSAL'
                            })
                    
                    # Shooting star (étoile filante)
                    elif upper_wick > 0.6 and body_ratio < 0.3 and lower_wick < 0.1:
                        # Contexte de tendance haussière
                        if i > 0 and candles[i-1]['close'] > candles[i-1]['open']:
                            patterns.append({
                                'type': 'SHOOTING_STAR',
                                'position': i,
                                'strength': upper_wick,
                                'confidence': 0.7,
                                'price_action': 'BEARISH_REVERSAL'
                            })
        
        return patterns
    
    def detect_doji(self, candles: List[Dict], lookback: int = 20) -> List[Dict]:
        """Détecte les doji"""
        patterns = []
        
        if len(candles) >= 1:
            for i in range(min(len(candles), lookback)):
                candle = candles[i]
                
                body = abs(candle['close'] - candle['open'])
                total_range = candle['high'] - candle['low']
                
                if total_range > 0:
                    body_ratio = body / total_range
                    
                    # Doji standard
                    if body_ratio < 0.1:
                        patterns.append({
                            'type': 'DOJI',
                            'position': i,
                            'strength': 1 - body_ratio,
                            'confidence': 0.6,
                            'price_action': 'INDECISION'
                        })
                    
                    # Long-legged doji
                    if body_ratio < 0.1 and total_range > np.mean([c['high'] - c['low'] for c in candles[max(0,i-5):i+1]]):
                        patterns.append({
                            'type': 'LONG_LEGGED_DOJI',
                            'position': i,
                            'strength': 0.8,
                            'confidence': 0.7,
                            'price_action': 'INDECISION'
                        })
        
        return patterns
    
    def detect_white_soldiers(self, candles: List[Dict], lookback: int = 20) -> List[Dict]:
        """Détecte les three white soldiers"""
        patterns = []
        
        if len(candles) >= 3:
            for i in range(2, min(len(candles), lookback)):
                if (candles[i-2]['close'] > candles[i-2]['open'] and
                    candles[i-1]['close'] > candles[i-1]['open'] and
                    candles[i]['close'] > candles[i]['open']):
                    
                    # Vérifier la progression
                    progression = (candles[i]['close'] > candles[i-1]['close'] > candles[i-2]['close'])
                    
                    if progression:
                        patterns.append({
                            'type': 'THREE_WHITE_SOLDIERS',
                            'position': i,
                            'strength': 0.8,
                            'confidence': 0.7,
                            'price_action': 'BULLISH_CONTINUATION'
                        })
        
        return patterns
    
    def detect_black_crows(self, candles: List[Dict], lookback: int = 20) -> List[Dict]:
        """Détecte les three black crows"""
        patterns = []
        
        if len(candles) >= 3:
            for i in range(2, min(len(candles), lookback)):
                if (candles[i-2]['close'] < candles[i-2]['open'] and
                    candles[i-1]['close'] < candles[i-1]['open'] and
                    candles[i]['close'] < candles[i]['open']):
                    
                    # Vérifier la progression
                    progression = (candles[i]['close'] < candles[i-1]['close'] < candles[i-2]['close'])
                    
                    if progression:
                        patterns.append({
                            'type': 'THREE_BLACK_CROWS',
                            'position': i,
                            'strength': 0.8,
                            'confidence': 0.7,
                            'price_action': 'BEARISH_CONTINUATION'
                        })
        
        return patterns
    
    def detect_morning_star(self, candles: List[Dict], lookback: int = 20) -> List[Dict]:
        """Détecte les morning star"""
        patterns = []
        
        if len(candles) >= 3:
            for i in range(2, min(len(candles), lookback)):
                first = candles[i-2]  # Grande bougie baissière
                second = candles[i-1]  # Petite bougie (doji ou petite)
                third = candles[i]     # Grande bougie haussière
                
                first_body = abs(first['close'] - first['open'])
                second_body = abs(second['close'] - second['open'])
                third_body = abs(third['close'] - third['open'])
                
                if (first['close'] < first['open'] and  # Première baissière
                    second_body < first_body * 0.5 and  # Seconde petite
                    third['close'] > third['open'] and  # Troisième haussière
                    third['close'] > first['open']):    # Clôture au-dessus du milieu de la première
                    
                    patterns.append({
                        'type': 'MORNING_STAR',
                        'position': i,
                        'strength': 0.8,
                        'confidence': 0.7,
                        'price_action': 'BULLISH_REVERSAL'
                    })
        
        return patterns
    
    def detect_evening_star(self, candles: List[Dict], lookback: int = 20) -> List[Dict]:
        """Détecte les evening star"""
        patterns = []
        
        if len(candles) >= 3:
            for i in range(2, min(len(candles), lookback)):
                first = candles[i-2]  # Grande bougie haussière
                second = candles[i-1]  # Petite bougie (doji ou petite)
                third = candles[i]     # Grande bougie baissière
                
                first_body = abs(first['close'] - first['open'])
                second_body = abs(second['close'] - second['open'])
                third_body = abs(third['close'] - third['open'])
                
                if (first['close'] > first['open'] and  # Première haussière
                    second_body < first_body * 0.5 and  # Seconde petite
                    third['close'] < third['open'] and  # Troisième baissière
                    third['close'] < first['open']):    # Clôture sous le milieu de la première
                    
                    patterns.append({
                        'type': 'EVENING_STAR',
                        'position': i,
                        'strength': 0.8,
                        'confidence': 0.7,
                        'price_action': 'BEARISH_REVERSAL'
                    })
        
        return patterns
    
    def detect_head_shoulders(self, candles: List[Dict], lookback: int = 50) -> List[Dict]:
        """Détecte les head and shoulders"""
        patterns = []
        
        if len(candles) >= 7:
            # Recherche simplifiée
            highs = [c['high'] for c in candles[-lookback:]]
            max_idx = np.argmax(highs)
            
            if 3 <= max_idx < len(highs) - 3:
                # Vérifier la structure H&S
                left_shoulder = np.max(highs[max_idx-3:max_idx-1]) if max_idx >= 3 else 0
                head = highs[max_idx]
                right_shoulder = np.max(highs[max_idx+1:max_idx+3]) if max_idx < len(highs) - 3 else 0
                
                # Vérifier la ligne de cou
                if (left_shoulder > 0 and right_shoulder > 0 and
                    left_shoulder < head and right_shoulder < head and
                    abs(left_shoulder - right_shoulder) < head * 0.01):  # 1% de tolérance
                    
                    patterns.append({
                        'type': 'HEAD_SHOULDERS',
                        'position': max_idx,
                        'strength': 0.7,
                        'confidence': 0.6,
                        'price_action': 'BEARISH_REVERSAL'
                    })
        
        return patterns
    
    def detect_inverse_head_shoulders(self, candles: List[Dict], lookback: int = 50) -> List[Dict]:
        """Détecte les inverse head and shoulders"""
        patterns = []
        
        if len(candles) >= 7:
            # Recherche simplifiée
            lows = [c['low'] for c in candles[-lookback:]]
            min_idx = np.argmin(lows)
            
            if 3 <= min_idx < len(lows) - 3:
                # Vérifier la structure inverse H&S
                left_shoulder = np.min(lows[min_idx-3:min_idx-1]) if min_idx >= 3 else 0
                head = lows[min_idx]
                right_shoulder = np.min(lows[min_idx+1:min_idx+3]) if min_idx < len(lows) - 3 else 0
                
                # Vérifier la ligne de cou
                if (left_shoulder > 0 and right_shoulder > 0 and
                    left_shoulder > head and right_shoulder > head and
                    abs(left_shoulder - right_shoulder) < head * 0.01):  # 1% de tolérance
                    
                    patterns.append({
                        'type': 'INVERSE_HEAD_SHOULDERS',
                        'position': min_idx,
                        'strength': 0.7,
                        'confidence': 0.6,
                        'price_action': 'BULLISH_REVERSAL'
                    })
        
        return patterns
    
    def detect_double_top(self, candles: List[Dict], lookback: int = 30) -> List[Dict]:
        """Détecte les double tops"""
        patterns = []
        
        if len(candles) >= 10:
            highs = [c['high'] for c in candles[-lookback:]]
            
            # Trouver les deux plus hauts pics
            peak_indices = []
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peak_indices.append(i)
            
            if len(peak_indices) >= 2:
                # Prendre les deux plus hauts pics
                peak_values = [highs[i] for i in peak_indices]
                sorted_indices = np.argsort(peak_values)[-2:]
                
                first_idx = peak_indices[sorted_indices[0]]
                second_idx = peak_indices[sorted_indices[1]]
                
                # Vérifier la proximité des prix
                if (first_idx < second_idx and
                    abs(highs[first_idx] - highs[second_idx]) < highs[second_idx] * 0.01):  # 1% de tolérance
                    
                    patterns.append({
                        'type': 'DOUBLE_TOP',
                        'position': second_idx,
                        'strength': 0.7,
                        'confidence': 0.6,
                        'price_action': 'BEARISH_REVERSAL'
                    })
        
        return patterns
    
    def detect_double_bottom(self, candles: List[Dict], lookback: int = 30) -> List[Dict]:
        """Détecte les double bottoms"""
        patterns = []
        
        if len(candles) >= 10:
            lows = [c['low'] for c in candles[-lookback:]]
            
            # Trouver les deux plus bas creux
            trough_indices = []
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    trough_indices.append(i)
            
            if len(trough_indices) >= 2:
                # Prendre les deux plus bas creux
                trough_values = [lows[i] for i in trough_indices]
                sorted_indices = np.argsort(trough_values)[:2]
                
                first_idx = trough_indices[sorted_indices[0]]
                second_idx = trough_indices[sorted_indices[1]]
                
                # Vérifier la proximité des prix
                if (first_idx < second_idx and
                    abs(lows[first_idx] - lows[second_idx]) < lows[second_idx] * 0.01):  # 1% de tolérance
                    
                    patterns.append({
                        'type': 'DOUBLE_BOTTOM',
                        'position': second_idx,
                        'strength': 0.7,
                        'confidence': 0.6,
                        'price_action': 'BULLISH_REVERSAL'
                    })
        
        return patterns
    
    # Méthodes simplifiées pour les autres patterns
    def detect_triangle(self, candles: List[Dict], lookback: int = 20) -> List[Dict]:
        """Détecte les triangles"""
        patterns = []
        return patterns
    
    def detect_wedge(self, candles: List[Dict], lookback: int = 20) -> List[Dict]:
        """Détecte les wedges"""
        patterns = []
        return patterns
    
    def detect_flag(self, candles: List[Dict], lookback: int = 15) -> List[Dict]:
        """Détecte les flags"""
        patterns = []
        return patterns
    
    def detect_pennant(self, candles: List[Dict], lookback: int = 15) -> List[Dict]:
        """Détecte les pennants"""
        patterns = []
        return patterns
    
    def calculate_pattern_score(self, patterns: List[Dict]) -> float:
        """Calcule un score global de pattern"""
        if not patterns:
            return 0.0
        
        # Poids par type de pattern
        pattern_weights = {
            'ENGULFING': 0.15,
            'HARAMI': 0.1,
            'HAMMER': 0.1,
            'DOJI': 0.05,
            'THREE_WHITE_SOLDIERS': 0.12,
            'THREE_BLACK_CROWS': 0.12,
            'MORNING_STAR': 0.15,
            'EVENING_STAR': 0.15,
            'HEAD_SHOULDERS': 0.08,
            'DOUBLE_TOP': 0.07,
            'DOUBLE_BOTTOM': 0.07
        }
        
        score = 0.0
        for pattern in patterns:
            pattern_type = pattern.get('type', '')
            base_type = pattern_type.split('_')[0] if '_' in pattern_type else pattern_type
            
            weight = pattern_weights.get(base_type, 0.05)
            strength = pattern.get('strength', 0.5)
            confidence = pattern.get('confidence', 0.5)
            
            score += weight * strength * confidence
        
        return min(score, 1.0)

class CycleDetector:
    """Détecteur de cycles de marché"""
    
    def detect(self, prices: List[float]) -> Dict:
        """Détecte les cycles dans les prix"""
        
        if len(prices) < 100:
            return {'cycles': [], 'dominant_cycle': None, 'cycle_score': 0.0}
        
        prices_np = np.array(prices)
        
        # 1. Transformée de Fourier pour les cycles
        fft_result = fft.fft(prices_np)
        frequencies = fft.fftfreq(len(prices_np))
        magnitudes = np.abs(fft_result)
        
        # 2. Trouver les fréquences dominantes (hors DC component)
        positive_freq = frequencies > 0
        dominant_indices = np.argsort(magnitudes[positive_freq])[-5:][::-1]
        
        cycles = []
        for idx in dominant_indices:
            freq = frequencies[positive_freq][idx]
            if freq > 0:
                period = 1 / freq if freq > 0 else 0
                magnitude = magnitudes[positive_freq][idx]
                
                cycles.append({
                    'period': float(period),
                    'frequency': float(freq),
                    'magnitude': float(magnitude),
                    'strength': float(magnitude / np.sum(magnitudes[positive_freq]))
                })
        
        # 3. Détection de cycles par autocorrélation
        autocorr_cycles = self.detect_autocorrelation_cycles(prices_np)
        cycles.extend(autocorr_cycles)
        
        # 4. Trouver le cycle dominant
        dominant_cycle = None
        if cycles:
            dominant_cycle = max(cycles, key=lambda x: x.get('strength', 0))
        
        return {
            'cycles': cycles,
            'dominant_cycle': dominant_cycle,
            'cycle_score': self.calculate_cycle_score(cycles),
            'cycle_stability': self.calculate_cycle_stability(prices_np, cycles)
        }
    
    def detect_autocorrelation_cycles(self, prices: np.ndarray, max_lag: int = 100) -> List[Dict]:
        """Détecte les cycles par autocorrélation"""
        cycles = []
        
        if len(prices) < max_lag * 2:
            return cycles
        
        # Calcul de l'autocorrélation
        autocorr = []
        for lag in range(1, min(max_lag, len(prices)//2)):
            if len(prices) > lag:
                corr = np.corrcoef(prices[:-lag], prices[lag:])[0, 1]
                autocorr.append(corr)
        
        # Trouver les pics d'autocorrélation (cycles)
        if len(autocorr) > 10:
            peaks, properties = find_peaks(autocorr, height=0.3, distance=5)
            
            for peak in peaks:
                cycles.append({
                    'period': float(peak + 1),
                    'method': 'autocorrelation',
                    'strength': float(autocorr[peak]),
                    'peak_position': int(peak)
                })
        
        return cycles
    
    def calculate_cycle_score(self, cycles: List[Dict]) -> float:
        """Calcule un score de cyclicité"""
        if not cycles:
            return 0.0
        
        # Score basé sur la force et la stabilité des cycles
        strengths = [c.get('strength', 0) for c in cycles]
        periods = [c.get('period', 0) for c in cycles]
        
        if strengths:
            avg_strength = np.mean(strengths)
            std_period = np.std(periods) if len(periods) > 1 else 0
            
            # Plus les cycles sont stables (faible std), meilleur est le score
            stability = 1.0 / (1.0 + std_period) if std_period > 0 else 1.0
            
            return float(avg_strength * stability)
        
        return 0.0
    
    def calculate_cycle_stability(self, prices: np.ndarray, cycles: List[Dict]) -> float:
        """Calcule la stabilité des cycles"""
        if not cycles or len(cycles) < 2:
            return 0.0
        
        # Vérifie la persistance des cycles dans le temps
        half_len = len(prices) // 2
        first_half = prices[:half_len]
        second_half = prices[half_len:]
        
        # Détecte les cycles dans chaque moitié
        first_cycles = self.detect(first_half)['cycles']
        second_cycles = self.detect(second_half)['cycles']
        
        if not first_cycles or not second_cycles:
            return 0.0
        
        # Compare les périodes dominantes
        first_periods = [c.get('period', 0) for c in first_cycles[:3]]
        second_periods = [c.get('period', 0) for c in second_cycles[:3]]
        
        # Calcul de la similarité
        similarity = 0.0
        for p1 in first_periods[:min(3, len(first_periods))]:
            for p2 in second_periods[:min(3, len(second_periods))]:
                if p1 > 0 and p2 > 0:
                    ratio = min(p1, p2) / max(p1, p2)
                    if ratio > 0.8:  # Périodes similaires à 20% près
                        similarity += 1
        
        max_pairs = min(3, len(first_periods), len(second_periods))
        return similarity / max_pairs if max_pairs > 0 else 0.0

class MomentumPredictor:
    """Prédicteur de momentum"""
    
    def predict(self, prices: List[float]) -> Dict:
        """Prédit le momentum futur"""
        
        if len(prices) < 20:
            return {'momentum': 0.0, 'direction': 'NEUTRAL', 'strength': 0.0, 'prediction': {}}
        
        prices_np = np.array(prices)
        
        # 1. Momentum classique
        momentum_short = self.calculate_momentum(prices_np, 5)
        momentum_medium = self.calculate_momentum(prices_np, 10)
        momentum_long = self.calculate_momentum(prices_np, 20)
        
        # 2. Dérivée du prix (accélération)
        acceleration = self.calculate_acceleration(prices_np)
        
        # 3. RSI pour le momentum relatif
        rsi = self.calculate_rsi(prices_np, 14)
        
        # 4. MACD pour le momentum directionnel
        macd, signal, hist = self.calculate_macd(prices_np)
        
        # 5. Force relative
        relative_strength = self.calculate_relative_strength(prices_np)
        
        # Synthèse
        momentum_score = (momentum_short + momentum_medium + momentum_long) / 3
        direction = 'BULLISH' if momentum_score > 0 else 'BEARISH' if momentum_score < 0 else 'NEUTRAL'
        
        # Prédiction à court terme
        prediction = self.predict_next_move(prices_np, momentum_score, acceleration)
        
        return {
            'momentum': float(momentum_score),
            'direction': direction,
            'strength': abs(float(momentum_score)),
            'acceleration': float(acceleration),
            'rsi': float(rsi[-1]) if len(rsi) > 0 else 50.0,
            'macd_hist': float(hist[-1]) if len(hist) > 0 else 0.0,
            'relative_strength': float(relative_strength),
            'prediction': prediction,
            'confidence': self.calculate_momentum_confidence(prices_np, momentum_score, acceleration)
        }
    
    def calculate_momentum(self, prices: np.ndarray, period: int) -> float:
        """Calcule le momentum sur une période"""
        if len(prices) < period:
            return 0.0
        return float((prices[-1] - prices[-period]) / prices[-period] * 100)  # En pourcentage
    
    def calculate_acceleration(self, prices: np.ndarray) -> float:
        """Calcule l'accélération (dérivée seconde)"""
        if len(prices) < 10:
            return 0.0
        
        # Dérivée première (vitesse)
        velocity = np.diff(prices[-10:])
        
        if len(velocity) < 5:
            return 0.0
        
        # Dérivée seconde (accélération)
        acceleration = np.diff(velocity[-5:])
        
        if len(acceleration) > 0:
            return float(np.mean(acceleration) * 10000)  # En pips
        return 0.0
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calcule le RSI"""
        if len(prices) < period + 1:
            return np.array([50.0])
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    def calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        """Calcule le MACD"""
        if len(prices) < slow + signal:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calcule l'EMA"""
        if len(data) < period:
            return np.zeros_like(data)
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[period-1] = np.mean(data[:period])
        
        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def calculate_relative_strength(self, prices: np.ndarray) -> float:
        """Calcule la force relative par rapport à la tendance"""
        if len(prices) < 30:
            return 0.5
        
        # Tendance à long terme
        long_trend = np.polyfit(range(30), prices[-30:], 1)[0]
        
        # Mouvement récent
        recent_move = prices[-1] - prices[-5]
        
        if abs(long_trend) > 0:
            # Force relative = mouvement récent / tendance attendue
            expected_move = long_trend * 5  # Sur 5 périodes
            relative_strength = recent_move / expected_move if expected_move != 0 else 0
            return float(relative_strength)
        
        return 0.5
    
    def predict_next_move(self, prices: np.ndarray, momentum: float, acceleration: float) -> Dict:
        """Prédit le prochain mouvement"""
        
        if len(prices) < 10:
            return {'direction': 'NEUTRAL', 'magnitude': 0.0, 'probability': 0.5}
        
        # Analyse des derniers mouvements
        recent_returns = np.diff(prices[-10:]) / prices[-11:-1]
        
        # Moyenne et écart-type des retours récents
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        # Probabilité de mouvement basée sur le momentum
        if momentum > 0:
            direction = 'UP'
            probability = 0.5 + min(abs(momentum) / 2, 0.3)  # Max 80%
            magnitude = mean_return + std_return * 0.5
        elif momentum < 0:
            direction = 'DOWN'
            probability = 0.5 + min(abs(momentum) / 2, 0.3)  # Max 80%
            magnitude = mean_return - std_return * 0.5
        else:
            direction = 'NEUTRAL'
            probability = 0.5
            magnitude = 0.0
        
        # Ajustement par accélération
        if acceleration > 0 and direction == 'UP':
            probability += 0.1
            magnitude *= 1.2
        elif acceleration < 0 and direction == 'DOWN':
            probability += 0.1
            magnitude *= 1.2
        
        return {
            'direction': direction,
            'magnitude': float(magnitude * 100),  # En pourcentage
            'probability': min(max(probability, 0.1), 0.9),
            'expected_pips': float(abs(magnitude) * prices[-1] * 10000)
        }
    
    def calculate_momentum_confidence(self, prices: np.ndarray, momentum: float, acceleration: float) -> float:
        """Calcule la confiance dans la prédiction de momentum"""
        
        if len(prices) < 20:
            return 0.5
        
        confidence_factors = []
        
        # 1. Consistance du momentum
        momentums = []
        for period in [3, 5, 8, 13]:
            if len(prices) >= period:
                mom = self.calculate_momentum(prices, period)
                momentums.append(mom)
        
        if momentums:
            momentum_consistency = 1.0 - (np.std(momentums) / (np.mean(np.abs(momentums)) + 1e-10))
            confidence_factors.append(min(momentum_consistency, 1.0))
        
        # 2. Volume du mouvement (si disponible, simplifié)
        price_range = np.ptp(prices[-10:])
        avg_range = np.mean([np.ptp(prices[-i-10:-i]) for i in range(5) if len(prices) >= i+10])
        
        if avg_range > 0:
            range_ratio = price_range / avg_range
            range_factor = min(range_ratio, 2.0) / 2.0  # Normalisé 0-1
            confidence_factors.append(range_factor)
        
        # 3. Accélération cohérente
        if abs(momentum) > 0:
            accel_factor = 1.0 - min(abs(acceleration / (momentum + 1e-10)), 1.0)
            confidence_factors.append(accel_factor)
        
        # 4. RSI confirmation
        rsi = self.calculate_rsi(prices, 14)
        if len(rsi) > 0:
            rsi_value = rsi[-1]
            if (momentum > 0 and rsi_value < 70) or (momentum < 0 and rsi_value > 30):
                rsi_factor = 0.8
            else:
                rsi_factor = 0.3
            confidence_factors.append(rsi_factor)
        
        return float(np.mean(confidence_factors)) if confidence_factors else 0.5

class ReversalPredictor:
    """Prédicteur de renversements"""
    
    def predict(self, prices: List[float]) -> Dict:
        """Prédit les potentiels de renversement"""
        
        if len(prices) < 30:
            return {'reversal_probability': 0.0, 'likely_direction': 'NONE', 'signals': []}
        
        prices_np = np.array(prices)
        
        signals = []
        
        # 1. Divergence RSI
        rsi_divergence = self.detect_rsi_divergence(prices_np)
        if rsi_divergence['detected']:
            signals.append(rsi_divergence)
        
        # 2. Surchauffe/ Survendu extrême
        extreme_conditions = self.detect_extreme_conditions(prices_np)
        if extreme_conditions['detected']:
            signals.append(extreme_conditions)
        
        # 3. Pattern de renversement
        reversal_patterns = self.detect_reversal_patterns(prices_np)
        signals.extend(reversal_patterns)
        
        # 4. Changement de momentum
        momentum_reversal = self.detect_momentum_reversal(prices_np)
        if momentum_reversal['detected']:
            signals.append(momentum_reversal)
        
        # 5. Test de support/résistance
        sr_test = self.detect_support_resistance_test(prices_np)
        if sr_test['detected']:
            signals.append(sr_test)
        
        # Calcul de la probabilité globale
        reversal_probability, likely_direction = self.calculate_reversal_probability(signals)
        
        return {
            'reversal_probability': reversal_probability,
            'likely_direction': likely_direction,
            'signals': signals,
            'signal_count': len(signals),
            'confidence': self.calculate_reversal_confidence(signals)
        }
    
    def detect_rsi_divergence(self, prices: np.ndarray, period: int = 14) -> Dict:
        """Détecte les divergences RSI"""
        
        if len(prices) < period * 2:
            return {'type': 'RSI_DIVERGENCE', 'detected': False}
        
        # Calcul du RSI
        rsi = self.calculate_rsi(prices, period)
        
        # Recherche de divergences
        lookback = min(30, len(prices) // 2)
        
        # Dernier pic/creux des prix
        price_extremes = self.find_price_extremes(prices[-lookback:])
        rsi_extremes = self.find_price_extremes(rsi[-lookback:])
        
        detected = False
        direction = 'NONE'
        strength = 0.0
        
        if price_extremes and rsi_extremes:
            # Vérifier la dernière divergence
            last_price_extreme = price_extremes[-1]
            last_rsi_extreme = rsi_extremes[-1]
            
            if (abs(last_price_extreme['position'] - last_rsi_extreme['position']) < 5 and
                last_price_extreme['type'] == last_rsi_extreme['type']):
                
                # Divergence régulière
                if (last_price_extreme['type'] == 'HIGH' and
                    last_price_extreme['value'] > price_extremes[-2]['value'] and
                    last_rsi_extreme['value'] < rsi_extremes[-2]['value']):
                    
                    detected = True
                    direction = 'BEARISH'
                    strength = 0.7
                
                elif (last_price_extreme['type'] == 'LOW' and
                      last_price_extreme['value'] < price_extremes[-2]['value'] and
                      last_rsi_extreme['value'] > rsi_extremes[-2]['value']):
                    
                    detected = True
                    direction = 'BULLISH'
                    strength = 0.7
        
        return {
            'type': 'RSI_DIVERGENCE',
            'detected': detected,
            'direction': direction,
            'strength': strength,
            'price_extremes': price_extremes,
            'rsi_extremes': rsi_extremes
        }
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calcule le RSI (identique à MomentumPredictor)"""
        if len(prices) < period + 1:
            return np.array([50.0])
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    def find_price_extremes(self, data: np.ndarray, window: int = 3) -> List[Dict]:
        """Trouve les pics et creux locaux"""
        extremes = []
        
        for i in range(window, len(data) - window):
            # Pic local
            if data[i] == np.max(data[i-window:i+window+1]):
                extremes.append({
                    'position': i,
                    'value': float(data[i]),
                    'type': 'HIGH'
                })
            # Creux local
            elif data[i] == np.min(data[i-window:i+window+1]):
                extremes.append({
                    'position': i,
                    'value': float(data[i]),
                    'type': 'LOW'
                })
        
        return extremes
    
    def detect_extreme_conditions(self, prices: np.ndarray) -> Dict:
        """Détecte les conditions de surchauffe/survendu"""
        
        if len(prices) < 20:
            return {'type': 'EXTREME_CONDITION', 'detected': False}
        
        # RSI
        rsi = self.calculate_rsi(prices, 14)
        current_rsi = rsi[-1] if len(rsi) > 0 else 50.0
        
        # Bandes de Bollinger
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(prices, 20)
        
        detected = False
        direction = 'NONE'
        strength = 0.0
        
        # Conditions extrêmes
        if current_rsi > 80:
            detected = True
            direction = 'BEARISH'
            strength = min((current_rsi - 80) / 20, 1.0)  # 0-1 scale
        
        elif current_rsi < 20:
            detected = True
            direction = 'BULLISH'
            strength = min((20 - current_rsi) / 20, 1.0)
        
        # Test des bandes de Bollinger
        current_price = prices[-1]
        if len(bb_upper) > 0 and len(bb_lower) > 0:
            bb_upper_val = bb_upper[-1]
            bb_lower_val = bb_lower[-1]
            
            if current_price > bb_upper_val:
                bb_strength = (current_price - bb_upper_val) / (bb_upper_val - bb_middle[-1])
                if bb_strength > 0.5:  > 50% hors de la bande
                    detected = True
                    direction = 'BEARISH'
                    strength = max(strength, min(bb_strength, 1.0))
            
            elif current_price < bb_lower_val:
                bb_strength = (bb_lower_val - current_price) / (bb_middle[-1] - bb_lower_val)
                if bb_strength > 0.5:
                    detected = True
                    direction = 'BULLISH'
                    strength = max(strength, min(bb_strength, 1.0))
        
        return {
            'type': 'EXTREME_CONDITION',
            'detected': detected,
            'direction': direction,
            'strength': strength,
            'rsi': float(current_rsi),
            'bb_position': float((current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])) if bb_upper[-1] != bb_lower[-1] else 0.5
        }
    
    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple:
        """Calcule les bandes de Bollinger"""
        if len(prices) < period:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
        
        sma = np.zeros_like(prices)
        std = np.zeros_like(prices)
        
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
            std[i] = np.std(prices[i-period+1:i+1])
        
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        
        return upper, sma, lower
    
    def detect_reversal_patterns(self, prices: np.ndarray) -> List[Dict]:
        """Détecte les patterns de renversement (simplifié)"""
        patterns = []
        
        if len(prices) < 10:
            return patterns
        
        # Double top/bottom simplifié
        lookback = min(30, len(prices))
        recent_prices = prices[-lookback:]
        
        # Trouver les extremums
        highs = []
        lows = []
        
        for i in range(2, len(recent_prices) - 2):
            if (recent_prices[i] > recent_prices[i-1] and
                recent_prices[i] > recent_prices[i-2] and
                recent_prices[i] > recent_prices[i+1] and
                recent_prices[i] > recent_prices[i+2]):
                highs.append((i, recent_prices[i]))
            
            if (recent_prices[i] < recent_prices[i-1] and
                recent_prices[i] < recent_prices[i-2] and
                recent_prices[i] < recent_prices[i+1] and
                recent_prices[i] < recent_prices[i+2]):
                lows.append((i, recent_prices[i]))
        
        # Vérifier les double tops
        if len(highs) >= 2:
            last_two_highs = highs[-2:]
            price_diff = abs(last_two_highs[0][1] - last_two_highs[1][1])
            avg_price = (last_two_highs[0][1] + last_two_highs[1][1]) / 2
            
            if price_diff < avg_price * 0.01:  # 1% de différence
                patterns.append({
                    'type': 'DOUBLE_TOP',
                    'detected': True,
                    'direction': 'BEARISH',
                    'strength': 0.7,
                    'highs': [(int(pos), float(val)) for pos, val in last_two_highs]
                })
        
        # Vérifier les double bottoms
        if len(lows) >= 2:
            last_two_lows = lows[-2:]
            price_diff = abs(last_two_lows[0][1] - last_two_lows[1][1])
            avg_price = (last_two_lows[0][1] + last_two_lows[1][1]) / 2
            
            if price_diff < avg_price * 0.01:  # 1% de différence
                patterns.append({
                    'type': 'DOUBLE_BOTTOM',
                    'detected': True,
                    'direction': 'BULLISH',
                    'strength': 0.7,
                    'lows': [(int(pos), float(val)) for pos, val in last_two_lows]
                })
        
        return patterns
    
    def detect_momentum_reversal(self, prices: np.ndarray) -> Dict:
        """Détecte les renversements de momentum"""
        
        if len(prices) < 15:
            return {'type': 'MOMENTUM_REVERSAL', 'detected': False}
        
        # Momentum sur différentes périodes
        momentum_3 = self.calculate_momentum(prices, 3)
        momentum_5 = self.calculate_momentum(prices, 5)
        momentum_8 = self.calculate_momentum(prices, 8)
        
        # Vérifier la convergence/divergence
        momentums = [momentum_3, momentum_5, momentum_8]
        momentums_sign = [1 if m > 0 else -1 if m < 0 else 0 for m in momentums]
        
        detected = False
        direction = 'NONE'
        strength = 0.0
        
        # Si les momentum s'inversent
        if (momentums_sign[0] != momentums_sign[1] or
            momentums_sign[1] != momentums_sign[2]):
            
            detected = True
            
            # Direction basée sur le momentum le plus court
            direction = 'BULLISH' if momentum_3 > 0 else 'BEARISH' if momentum_3 < 0 else 'NONE'
            
            # Force basée sur la divergence
            strength = abs(momentum_3 - momentum_8) / (abs(momentum_8) + 1e-10)
            strength = min(strength, 1.0)
        
        return {
            'type': 'MOMENTUM_REVERSAL',
            'detected': detected,
            'direction': direction,
            'strength': strength,
            'momentums': {
                '3_period': float(momentum_3),
                '5_period': float(momentum_5),
                '8_period': float(momentum_8)
            }
        }
    
    def calculate_momentum(self, prices: np.ndarray, period: int) -> float:
        """Calcule le momentum (identique à MomentumPredictor)"""
        if len(prices) < period:
            return 0.0
        return float((prices[-1] - prices[-period]) / prices[-period] * 100)
    
    def detect_support_resistance_test(self, prices: np.ndarray) -> Dict:
        """Détecte les tests de support/résistance"""
        
        if len(prices) < 20:
            return {'type': 'SUPPORT_RESISTANCE_TEST', 'detected': False}
        
        # Identifier les niveaux de support/résistance
        levels = self.identify_support_resistance(prices[-50:] if len(prices) >= 50 else prices)
        
        current_price = prices[-1]
        detected = False
        direction = 'NONE'
        strength = 0.0
        tested_level = None
        
        for level in levels:
            distance = abs(current_price - level['price'])
            
            # Si proche d'un niveau (dans 0.5%)
            if distance < current_price * 0.005:
                detected = True
                tested_level = level
                
                # Déterminer la direction
                if level['type'] == 'RESISTANCE':
                    direction = 'BEARISH'
                else:  # SUPPORT
                    direction = 'BULLISH'
                
                # Force basée sur la proximité et la force du niveau
                proximity = 1.0 - (distance / (current_price * 0.005))
                strength = level['strength'] * proximity
                
                break
        
        return {
            'type': 'SUPPORT_RESISTANCE_TEST',
            'detected': detected,
            'direction': direction,
            'strength': strength,
            'tested_level': tested_level,
            'current_price': float(current_price)
        }
    
    def identify_support_resistance(self, prices: np.ndarray) -> List[Dict]:
        """Identifie les niveaux de support et résistance"""
        levels = []
        
        if len(prices) < 10:
            return levels
        
        # Utiliser les pivots et les clusters de prix
        window = 5
        
        for i in range(window, len(prices) - window):
            # Résistance locale
            if prices[i] == np.max(prices[i-window:i+window+1]):
                # Vérifier si ce niveau a été testé plusieurs fois
                touch_count = 0
                for j in range(len(prices)):
                    if abs(prices[j] - prices[i]) < prices[i] * 0.002:  # 0.2%
                        touch_count += 1
                
                if touch_count >= 2:
                    levels.append({
                        'price': float(prices[i]),
                        'type': 'RESISTANCE',
                        'strength': min(touch_count / 5, 1.0),
                        'position': i
                    })
            
            # Support local
            elif prices[i] == np.min(prices[i-window:i+window+1]):
                # Vérifier si ce niveau a été testé plusieurs fois
                touch_count = 0
                for j in range(len(prices)):
                    if abs(prices[j] - prices[i]) < prices[i] * 0.002:  # 0.2%
                        touch_count += 1
                
                if touch_count >= 2:
                    levels.append({
                        'price': float(prices[i]),
                        'type': 'SUPPORT',
                        'strength': min(touch_count / 5, 1.0),
                        'position': i
                    })
        
        # Trier par force
        levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return levels[:5]  # Retourner les 5 niveaux les plus forts
    
    def calculate_reversal_probability(self, signals: List[Dict]) -> Tuple[float, str]:
        """Calcule la probabilité globale de renversement"""
        
        if not signals:
            return 0.0, 'NONE'
        
        # Poids par type de signal
        signal_weights = {
            'RSI_DIVERGENCE': 0.25,
            'EXTREME_CONDITION': 0.20,
            'DOUBLE_TOP': 0.15,
            'DOUBLE_BOTTOM': 0.15,
            'MOMENTUM_REVERSAL': 0.15,
            'SUPPORT_RESISTANCE_TEST': 0.10
        }
        
        bullish_score = 0.0
        bearish_score = 0.0
        
        for signal in signals:
            if signal.get('detected', False):
                signal_type = signal.get('type', '')
                direction = signal.get('direction', 'NONE')
                strength = signal.get('strength', 0.0)
                weight = signal_weights.get(signal_type, 0.05)
                
                signal_score = weight * strength
                
                if direction == 'BULLISH':
                    bullish_score += signal_score
                elif direction == 'BEARISH':
                    bearish_score += signal_score
        
        # Probabilité et direction
        if bullish_score > bearish_score:
            probability = bullish_score
            direction = 'BULLISH'
        elif bearish_score > bullish_score:
            probability = bearish_score
            direction = 'BEARISH'
        else:
            probability = max(bullish_score, bearish_score)
            direction = 'NONE'
        
        return min(probability, 1.0), direction
    
    def calculate_reversal_confidence(self, signals: List[Dict]) -> float:
        """Calcule la confiance dans la prédiction de renversement"""
        
        if not signals:
            return 0.0
        
        # Facteurs de confiance
        confidence_factors = []
        
        # 1. Nombre de signaux concordants
        bullish_count = sum(1 for s in signals if s.get('direction') == 'BULLISH' and s.get('detected', False))
        bearish_count = sum(1 for s in signals if s.get('direction') == 'BEARISH' and s.get('detected', False))
        
        if bullish_count > 0 or bearish_count > 0:
            agreement = max(bullish_count, bearish_count) / (bullish_count + bearish_count)
            confidence_factors.append(agreement)
        
        # 2. Force moyenne des signaux
        strengths = [s.get('strength', 0) for s in signals if s.get('detected', False)]
        if strengths:
            avg_strength = np.mean(strengths)
            confidence_factors.append(avg_strength)
        
        # 3. Diversité des signaux
        signal_types = set(s.get('type', '') for s in signals if s.get('detected', False))
        diversity = len(signal_types) / 6  # 6 types maximum
        confidence_factors.append(diversity)
        
        return float(np.mean(confidence_factors)) if confidence_factors else 0.0

class VolatilityPredictor:
    """Prédicteur de volatilité"""
    
    def predict(self, prices: List[float]) -> Dict:
        """Prédit la volatilité future"""
        
        if len(prices) < 20:
            return {'volatility_forecast': 0.0, 'regime': 'LOW', 'confidence': 0.5, 'breakout_probability': 0.0}
        
        prices_np = np.array(prices)
        returns = np.diff(prices_np) / prices_np[:-1]
        
        # 1. Volatilité historique
        historical_vol = self.calculate_historical_volatility(returns)
        
        # 2. Volatilité implicite (simplifiée - basée sur les extrêmes récents)
        implied_vol = self.estimate_implied_volatility(prices_np)
        
        # 3. Modèle GARCH simplifié
        garch_vol = self.garch_prediction(returns)
        
        # 4. Régime de volatilité
        volatility_regime = self.determine_volatility_regime(historical_vol, implied_vol)
        
        # 5. Probabilité de breakout
        breakout_prob = self.calculate_breakout_probability(prices_np, historical_vol)
        
        # Prévision combinée
        volatility_forecast = (historical_vol * 0.4 + implied_vol * 0.3 + garch_vol * 0.3)
        
        return {
            'volatility_forecast': float(volatility_forecast),
            'historical_volatility': float(historical_vol),
            'implied_volatility': float(implied_vol),
            'garch_volatility': float(garch_vol),
            'regime': volatility_regime,
            'breakout_probability': float(breakout_prob),
            'confidence': self.calculate_volatility_confidence(historical_vol, implied_vol, garch_vol),
            'volatility_clusters': self.detect_volatility_clusters(returns)
        }
    
    def calculate_historical_volatility(self, returns: np.ndarray, annualize: bool = True) -> float:
        """Calcule la volatilité historique"""
        if len(returns) < 10:
            return 0.0
        
        # Volatilité sur différentes périodes
        volatilities = []
        for period in [5, 10, 20]:
            if len(returns) >= period:
                period_returns = returns[-period:]
                vol = np.std(period_returns)
                if annualize:
                    vol *= np.sqrt(252)  # Annualisation (jours de trading)
                volatilities.append(vol)
        
        # Moyenne pondérée (plus de poids aux périodes récentes)
        if volatilities:
            weights = [0.2, 0.3, 0.5]  # Plus de poids à la période la plus longue
            weighted_vol = sum(v * w for v, w in zip(volatilities, weights[:len(volatilities)]))
            return float(weighted_vol)
        
        return 0.0
    
    def estimate_implied_volatility(self, prices: np.ndarray) -> float:
        """Estime la volatilité implicite (simplifiée)"""
        if len(prices) < 20:
            return 0.0
        
        # Basé sur la range récente
        recent_range = np.ptp(prices[-10:])
        avg_range = np.mean([np.ptp(prices[-i-10:-i]) for i in range(5) if len(prices) >= i+10])
        
        if avg_range > 0:
            range_ratio = recent_range / avg_range
            # Volatilité implicite estimée
            base_vol = np.std(np.diff(prices[-20:]) / prices[-21:-1])
            implied_vol = base_vol * range_ratio * np.sqrt(252)  # Annualisé
            return float(implied_vol)
        
        return 0.0
    
    def garch_prediction(self, returns: np.ndarray, p: int = 1, q: int = 1) -> float:
        """Prédiction GARCH simplifiée"""
        if len(returns) < 30:
            return 0.0
        
        # Modèle GARCH(1,1) simplifié
        omega = 0.05
        alpha = 0.1
        beta = 0.85
        
        # Variance initiale
        variance = np.var(returns[-30:])
        
        # Mise à jour GARCH
        last_return = returns[-1]
        variance = omega + alpha * last_return**2 + beta * variance
        
        return float(np.sqrt(variance) * np.sqrt(252))  # Annualisé
    
    def determine_volatility_regime(self, historical_vol: float, implied_vol: float) -> str:
        """Détermine le régime de volatilité"""
        
        # Seuls (en volatilité annualisée)
        if historical_vol < 0.10:  # 10%
            regime = "VERY_LOW"
        elif historical_vol < 0.15:  # 15%
            regime = "LOW"
        elif historical_vol < 0.25:  # 25%
            regime = "NORMAL"
        elif historical_vol < 0.40:  # 40%
            regime = "HIGH"
        else:
            regime = "VERY_HIGH"
        
        # Ajustement basé sur la volatilité implicite
        vol_ratio = implied_vol / historical_vol if historical_vol > 0 else 1.0
        
        if vol_ratio > 1.5:
            regime += "_EXPECTING_INCREASE"
        elif vol_ratio < 0.7:
            regime += "_EXPECTING_DECREASE"
        
        return regime
    
    def calculate_breakout_probability(self, prices: np.ndarray, volatility: float) -> float:
        """Calcule la probabilité de breakout"""
        if len(prices) < 20 or volatility == 0:
            return 0.0
        
        # Analyse de compression
        recent_range = np.ptp(prices[-10:])
        historical_range = np.mean([np.ptp(prices[-i-20:-i]) for i in range(10) if len(prices) >= i+20])
        
        if historical_range > 0:
            compression_ratio = recent_range / historical_range
            
            # Plus la compression est forte, plus la probabilité de breakout est élevée
            if compression_ratio < 0.5:
                probability = 0.7
            elif compression_ratio < 0.7:
                probability = 0.5
            elif compression_ratio < 0.9:
                probability = 0.3
            else:
                probability = 0.1
            
            # Ajustement par volatilité
            volatility_adjustment = min(volatility / 0.2, 1.0)  # Normalisé à 20% vol
            probability *= volatility_adjustment
            
            return float(probability)
        
        return 0.0
    
    def calculate_volatility_confidence(self, hist_vol: float, impl_vol: float, garch_vol: float) -> float:
        """Calcule la confiance dans la prédiction de volatilité"""
        
        # Consistance entre les différentes mesures
        vols = [hist_vol, impl_vol, garch_vol]
        valid_vols = [v for v in vols if v > 0]
        
        if len(valid_vols) < 2:
            return 0.5
        
        # Coefficient de variation (plus faible = plus confiant)
        cv = np.std(valid_vols) / np.mean(valid_vols)
        consistency = 1.0 / (1.0 + cv)
        
        # Confiance basée sur la quantité de données
        data_confidence = min(len(valid_vols) / 3, 1.0)
        
        return float((consistency + data_confidence) / 2)
    
    def detect_volatility_clusters(self, returns: np.ndarray) -> List[Dict]:
        """Détecte les clusters de volatilité"""
        clusters = []
        
        if len(returns) < 30:
            return clusters
        
        # Détection de changements de régime de volatilité
        window = 10
        volatilities = []
        
        for i in range(window, len(returns)):
            vol = np.std(returns[i-window:i])
            volatilities.append(vol)
        
        if len(volatilities) > 10:
            # Détection de clusters (périodes de volatilité élevée)
            mean_vol = np.mean(volatilities)
            std_vol = np.std(volatilities)
            
            high_vol_indices = np.where(np.array(volatilities) > mean_vol + std_vol)[0]
            
            if len(high_vol_indices) > 0:
                # Regrouper les indices consécutifs
                clusters = []
                current_cluster = [high_vol_indices[0]]
                
                for i in range(1, len(high_vol_indices)):
                    if high_vol_indices[i] == high_vol_indices[i-1] + 1:
                        current_cluster.append(high_vol_indices[i])
                    else:
                        if len(current_cluster) >= 3:  # Minimum 3 périodes
                            clusters.append({
                                'start': int(current_cluster[0]),
                                'end': int(current_cluster[-1]),
                                'duration': len(current_cluster),
                                'avg_volatility': float(np.mean([volatilities[j] for j in current_cluster])),
                                'peak_volatility': float(np.max([volatilities[j] for j in current_cluster]))
                            })
                        current_cluster = [high_vol_indices[i]]
                
                # Dernier cluster
                if len(current_cluster) >= 3:
                    clusters.append({
                        'start': int(current_cluster[0]),
                        'end': int(current_cluster[-1]),
                        'duration': len(current_cluster),
                        'avg_volatility': float(np.mean([volatilities[j] for j in current_cluster])),
                        'peak_volatility': float(np.max([volatilities[j] for j in current_cluster]))
                    })
        
        return clusters

# ============================================================================
# FILTRES ET LISSAGES
# ============================================================================

class KalmanFilter:
    """Filtre de Kalman pour le lissage des prix"""
    
    def __init__(self):
        self.x = 0.0  # État estimé
        self.P = 1.0  # Covariance de l'erreur
        self.Q = 0.01  # Bruit de processus
        self.R = 0.1   # Bruit de mesure
    
    def filter(self, prices: List[float]) -> Dict:
        """Applique le filtre de Kalman"""
        
        if len(prices) < 10:
            return {'filtered': prices, 'trend': 0.0, 'uncertainty': 0.0}
        
        filtered_prices = []
        trends = []
        uncertainties = []
        
        # Réinitialiser pour chaque nouvelle série
        self.x = prices[0]
        self.P = 1.0
        
        for price in prices:
            # Prédiction
            x_pred = self.x
            P_pred = self.P + self.Q
            
            # Mise à jour
            K = P_pred / (P_pred + self.R)  # Gain de Kalman
            self.x = x_pred + K * (price - x_pred)
            self.P = (1 - K) * P_pred
            
            filtered_prices.append(self.x)
            trends.append(K * (price - x_pred))  # Tendance estimée
            uncertainties.append(self.P)
        
        return {
            'filtered': [float(p) for p in filtered_prices],
            'trend': float(trends[-1]) if trends else 0.0,
            'uncertainty': float(uncertainties[-1]) if uncertainties else 0.0,
            'kalman_gain': float(K) if 'K' in locals() else 0.0,
            'filter_convergence': self.calculate_convergence(uncertainties)
        }
    
    def calculate_convergence(self, uncertainties: List[float]) -> float:
        """Calcule le degré de convergence du filtre"""
        if len(uncertainties) < 10:
            return 0.0
        
        # Réduction relative de l'incertitude
        initial_uncertainty = uncertainties[0]
        final_uncertainty = uncertainties[-1]
        
        if initial_uncertainty > 0:
            convergence = 1.0 - (final_uncertainty / initial_uncertainty)
            return float(max(0.0, min(convergence, 1.0)))
        
        return 0.0

class BandpassFilter:
    """Filtre passe-bande pour isoler les fréquences spécifiques"""
    
    def __init__(self, lowcut: float = 0.01, highcut: float = 0.5, order: int = 3):
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
    
    def filter(self, prices: List[float], fs: float = 1.0) -> Dict:
        """Applique le filtre passe-bande"""
        
        if len(prices) < 50:
            return {'filtered': prices, 'frequency_response': {}, 'band_energy': 0.0}
        
        prices_np = np.array(prices)
        
        try:
            # Créer le filtre butterworth
            nyquist = 0.5 * fs
            low = self.lowcut / nyquist
            high = self.highcut / nyquist
            
            b, a = butter(self.order, [low, high], btype='band')
            filtered = filtfilt(b, a, prices_np)
            
            # Réponse en fréquence
            freqs, response = signal.freqz(b, a, worN=2000)
            
            # Énergie dans la bande passante
            band_energy = np.sum(filtered**2) / len(filtered)
            
            return {
                'filtered': [float(p) for p in filtered],
                'frequency_response': {
                    'frequencies': [float(f) for f in freqs],
                    'magnitudes': [float(abs(r)) for r in response]
                },
                'band_energy': float(band_energy),
                'signal_to_noise': self.calculate_snr(prices_np, filtered),
                'dominant_frequency': self.find_dominant_frequency(filtered, fs)
            }
        except:
            return {'filtered': prices, 'frequency_response': {}, 'band_energy': 0.0}
    
    def calculate_snr(self, original: np.ndarray, filtered: np.ndarray) -> float:
        """Calcule le ratio signal/bruit"""
        if len(original) != len(filtered):
            return 0.0
        
        noise = original - filtered
        signal_power = np.mean(filtered**2)
        noise_power = np.mean(noise**2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return float(snr)
        
        return 0.0
    
    def find_dominant_frequency(self, signal: np.ndarray, fs: float) -> float:
        """Trouve la fréquence dominante dans le signal filtré"""
        if len(signal) < 20:
            return 0.0
        
        fft_result = fft.fft(signal)
        frequencies = fft.fftfreq(len(signal), 1/fs)
        
        positive = frequencies > 0
        magnitudes = np.abs(fft_result[positive])
        
        if len(magnitudes) > 0:
            dominant_idx = np.argmax(magnitudes)
            dominant_freq = frequencies[positive][dominant_idx]
            return float(dominant_freq)
        
        return 0.0

class NoiseFilter:
    """Filtre de bruit avancé"""
    
    def filter(self, prices: List[float]) -> Dict:
        """Filtre le bruit des prix"""
        
        if len(prices) < 20:
            return {'denoised': prices, 'noise_level': 0.0, 'signal_purity': 0.0}
        
        prices_np = np.array(prices)
        
        # 1. Filtre Savitzky-Golay
        window = min(15, len(prices_np) // 2)
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            savgol_filtered = savgol_filter(prices_np, window, 3)
        else:
            savgol_filtered = prices_np
        
        # 2. Filtre médian
        window = min(5, len(prices_np))
        median_filtered = signal.medfilt(prices_np, kernel_size=window)
        
        # 3. Combinaison adaptative
        denoised = self.adaptive_combination(prices_np, savgol_filtered, median_filtered)
        
        # Analyse du bruit
        noise = prices_np - denoised
        noise_level = np.std(noise) / np.std(prices_np) if np.std(prices_np) > 0 else 0
        
        return {
            'denoised': [float(p) for p in denoised],
            'noise_level': float(noise_level),
            'signal_purity': float(1.0 - noise_level),
            'noise_characteristics': self.analyze_noise(noise),
            'filter_performance': self.evaluate_filter_performance(prices_np, denoised)
        }
    
    def adaptive_combination(self, original: np.ndarray, savgol: np.ndarray, median: np.ndarray) -> np.ndarray:
        """Combine adaptativement les différents filtres"""
        
        # Mesure de la volatilité locale
        volatility = np.std(np.diff(original[-10:])) if len(original) >= 11 else 0
        
        # Poids adaptatifs
        if volatility < 0.001:  # Très calme
            weight_savgol = 0.7
            weight_median = 0.3
        elif volatility < 0.005:  # Calme
            weight_savgol = 0.5
            weight_median = 0.5
        else:  # Volatil
            weight_savgol = 0.3
            weight_median = 0.7
        
        # Combinaison
        combined = weight_savgol * savgol + weight_median * median
        
        # Assurer la continuité
        combined = self.ensure_continuity(combined, original)
        
        return combined
    
    def ensure_continuity(self, filtered: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Assure la continuité du signal filtré"""
        # Ajuster pour éviter les discontinuités
        diff = filtered - original
        smooth_diff = savgol_filter(diff, min(7, len(diff)), 2)
        
        return original + smooth_diff
    
    def analyze_noise(self, noise: np.ndarray) -> Dict:
        """Analyse les caractéristiques du bruit"""
        if len(noise) < 10:
            return {'distribution': 'UNKNOWN', 'autocorrelation': 0.0, 'whiteness': 0.0}
        
        # Distribution
        skewness = skew(noise)
        kurt = kurtosis(noise)
        
        if abs(skewness) < 0.5 and abs(kurt) < 1:
            distribution = 'GAUSSIAN'
        elif skewness > 0.5:
            distribution = 'POSITIVE_SKEW'
        elif skewness < -0.5:
            distribution = 'NEGATIVE_SKEW'
        else:
            distribution = 'NON_GAUSSIAN'
        
        # Autocorrélation
        if len(noise) > 5:
            autocorr = np.corrcoef(noise[:-1], noise[1:])[0, 1]
        else:
            autocorr = 0.0
        
        # Test de blancheur (simplifié)
        whiteness = 1.0 - abs(autocorr)
        
        return {
            'distribution': distribution,
            'skewness': float(skewness),
            'kurtosis': float(kurt),
            'autocorrelation': float(autocorr),
            'whiteness': float(whiteness),
            'noise_power': float(np.mean(noise**2))
        }
    
    def evaluate_filter_performance(self, original: np.ndarray, filtered: np.ndarray) -> Dict:
        """Évalue la performance du filtre"""
        
        noise = original - filtered
        
        # SNR amélioration
        original_snr = np.mean(original**2) / (np.var(original) + 1e-10)
        filtered_snr = np.mean(filtered**2) / (np.var(noise) + 1e-10)
        
        snr_improvement = filtered_snr / original_snr if original_snr > 0 else 1.0
        
        # Préservation du signal
        correlation = np.corrcoef(original, filtered)[0, 1]
        
        # Smoothness
        original_roughness = np.mean(np.abs(np.diff(original)))
        filtered_roughness = np.mean(np.abs(np.diff(filtered)))
        smoothness_ratio = original_roughness / (filtered_roughness + 1e-10)
        
        return {
            'snr_improvement': float(snr_improvement),
            'signal_preservation': float(correlation),
            'smoothness_ratio': float(smoothness_ratio),
            'noise_reduction': float(1.0 - (np.var(noise) / np.var(original)) if np.var(original) > 0 else 0.0)
        }

# ============================================================================
# SYSTÈME DE DÉCISION QUANTIQUE
# ============================================================================

class QuantumDecisionSystem:
    """SYSTÈME DE DÉCISION QUANTIQUE AVANCÉ"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.quantum_engine = QuantumEngine(config)
        self.learning_memory = LearningMemory()
        self.decision_models = {}
        self.initialize_models()
        
        # État interne
        self.current_minute = None
        self.scalps_this_minute = 0
        self.scalp_success_rates = deque(maxlen=100)
        self.performance_tracker = PerformanceTracker()
        
        # Optimisation dynamique
        self.parameter_optimizer = ParameterOptimizer()
        self.strategy_evolver = StrategyEvolver()
        
    def initialize_models(self):
        """Initialise les modèles de décision"""
        
        # Modèle de classification sociale
        self.decision_models['social_classifier'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Modèle de timing
        self.decision_models['timing_model'] = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Modèle de direction
        self.decision_models['direction_model'] = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        
        # Réseau neuronal profond
        self.decision_models['neural_model'] = self.build_neural_network()
        
        # Modèle de risque
        self.decision_models['risk_model'] = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=4,
            random_state=42
        )
    
    def build_neural_network(self):
        """Construit le réseau neuronal profond"""
        model = models.Sequential([
            layers.Dense(self.config.neural_layers[0], activation='relu', 
                        input_shape=(50,)),
            layers.Dropout(self.config.dropout_rate),
            layers.Dense(self.config.neural_layers[1], activation='relu'),
            layers.Dropout(self.config.dropout_rate),
            layers.Dense(self.config.neural_layers[2], activation='relu'),
            layers.Dense(self.config.neural_layers[3], activation='relu'),
            layers.Dense(3, activation='softmax')  # BUY, SELL, HOLD
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def make_decision(self, market_data: Dict, current_time: datetime) -> Dict:
        """Prend une décision de trading quantique"""
        
        # Vérifier le timing de scalp
        current_minute = current_time.replace(second=0, microsecond=0)
        if current_minute != self.current_minute:
            self.current_minute = current_minute
            self.scalps_this_minute = 0
        
        # Vérifier les limites de scalp
        if self.scalps_this_minute >= self.config.scalp_per_minute_target:
            return {
                'action': 'HOLD',
                'reason': 'Scalp limit reached for this minute',
                'confidence': 0.0,
                'scalp_opportunity': False
            }
        
        # Analyse quantique complète
        quantum_analysis = self.quantum_engine.analyze_tick_stream(
            market_data.get('tick_stream', [])
        )
        
        if not quantum_analysis:
            return {
                'action': 'HOLD',
                'reason': 'Insufficient data for quantum analysis',
                'confidence': 0.0,
                'scalp_opportunity': False
            }
        
        # Extraction de features
        features = self.extract_decision_features(market_data, quantum_analysis)
        
        # Prédictions des modèles
        predictions = self.get_model_predictions(features)
        
        # Synthèse quantique des décisions
        final_decision = self.synthesize_decisions(predictions, features, current_time)
        
        # Vérification des opportunités de scalp
        scalp_opportunity = self.is_scalp_opportunity(current_time, final_decision)
        
        if scalp_opportunity and final_decision['action'] != 'HOLD':
            self.scalps_this_minute += 1
            
            return {
                'action': final_decision['action'],
                'direction': final_decision['action'],  # BUY/SELL
                'confidence': final_decision['confidence'],
                'reason': f"{final_decision['reason']} | Scalp #{self.scalps_this_minute} this minute",
                'scalp_opportunity': True,
                'scalp_number': self.scalps_this_minute,
                'quantum_state': quantum_analysis.get('quantum_synthesis', {}).get('quantum_state', 'UNKNOWN'),
                'social_class': features.get('social_class', 'UNKNOWN'),
                'risk_score': final_decision.get('risk_score', 0.5),
                'lot_size': self.calculate_lot_size(final_decision['confidence'], features),
                'take_profit': self.calculate_take_profit(features),
                'stop_loss': self.calculate_stop_loss(features),
                'scalp_duration': self.calculate_scalp_duration(features),
                'features': features,
                'predictions': predictions
            }
        
        else:
            return {
                'action': 'HOLD',
                'reason': 'No scalp opportunity identified',
                'confidence': 0.0,
                'scalp_opportunity': False,
                'quantum_state': quantum_analysis.get('quantum_synthesis', {}).get('quantum_state', 'UNKNOWN')
            }
    
    def extract_decision_features(self, market_data: Dict, quantum_analysis: Dict) -> Dict:
        """Extrait les features pour la décision"""
        
        features = {}
        
        # Prix et volume
        if market_data.get('tick_stream'):
            ticks = market_data['tick_stream']
            if ticks:
                last_tick = ticks[-1]
                features['current_price'] = last_tick.get('bid', 0)
                features['current_spread'] = (last_tick.get('ask', 0) - last_tick.get('bid', 0)) * 10000
                features['current_volume'] = last_tick.get('volume', 0)
        
        # Analyse quantique
        quantum_synth = quantum_analysis.get('quantum_synthesis', {})
        features['quantum_state'] = quantum_synth.get('quantum_state', 'UNKNOWN')
        features['quantum_confidence'] = quantum_synth.get('state_confidence', 0.0)
        
        # Scores quantiques
        quantum_scores = quantum_synth.get('scores', {})
        for key, value in quantum_scores.items():
            features[f'quantum_{key}'] = value
        
        # Analyses détaillées
        analyses = quantum_analysis.get('analyses', {})
        
        if 'microstructure' in analyses:
            micro = analyses['microstructure']
            features['volatility'] = micro.get('volatility', 0)
            features['entropy'] = micro.get('entropy', 0)
            features['hurst_exponent'] = micro.get('hurst_exponent', 0.5)
        
        if 'wavelet' in analyses:
            wavelet = analyses['wavelet']
            features['wavelet_energy'] = wavelet.get('scale_energy', 0)
        
        if 'hilbert' in analyses:
            hilbert = analyses['hilbert']
            features['hilbert_coherence'] = hilbert.get('coherence', 0)
        
        if 'social' in analyses:
            social = analyses['social']
            features['social_class'] = social.get('class', 'UNKNOWN')
            features['social_confidence'] = social.get('confidence', 0.0)
        
        # Patterns et anomalies
        if 'anomalies' in analyses:
            anomalies = analyses['anomalies']
            features['anomaly_score'] = anomalies.get('anomaly_score', 0)
            features['opportunity_score'] = anomalies.get('opportunity_score', 0)
        
        if 'patterns' in analyses:
            patterns = analyses['patterns']
            features['pattern_score'] = patterns.get('pattern_score', 0)
        
        # Momentum et reversal
        if 'momentum' in analyses:
            momentum = analyses['momentum']
            features['momentum_direction'] = momentum.get('direction', 'NEUTRAL')
            features['momentum_strength'] = momentum.get('strength', 0)
            features['momentum_confidence'] = momentum.get('confidence', 0.5)
        
        if 'reversal' in analyses:
            reversal = analyses['reversal']
            features['reversal_probability'] = reversal.get('reversal_probability', 0)
            features['reversal_direction'] = reversal.get('likely_direction', 'NONE')
        
        # Volatility
        if 'volatility' in analyses:
            volatility = analyses['volatility']
            features['volatility_forecast'] = volatility.get('volatility_forecast', 0)
            features['volatility_regime'] = volatility.get('regime', 'NORMAL')
        
        # Facteur temps
        current_time = datetime.now()
        features['minute_of_hour'] = current_time.minute
        features['second_of_minute'] = current_time.second
        features['hour_of_day'] = current_time.hour
        
        # Scalps précédents ce minute
        features['scalps_this_minute'] = self.scalps_this_minute
        
        return features
    
    def get_model_predictions(self, features: Dict) -> Dict:
        """Obtient les prédictions de tous les modèles"""
        
        predictions = {}
        
        # Préparation des données d'entrée
        feature_vector = self.prepare_feature_vector(features)
        
        # Prédictions de chaque modèle
        for model_name, model in self.decision_models.items():
            try:
                if model_name == 'neural_model':
                    # Pour le réseau neuronal
                    if len(feature_vector) == 50:
                        input_data = np.array([feature_vector])
                        pred = model.predict(input_data, verbose=0)[0]
                        predictions[model_name] = {
                            'BUY': float(pred[0]),
                            'SELL': float(pred[1]),
                            'HOLD': float(pred[2])
                        }
                else:
                    # Pour les autres modèles
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba([feature_vector])[0]
                        if len(pred_proba) == 3:  # BUY, SELL, HOLD
                            predictions[model_name] = {
                                'BUY': float(pred_proba[0]),
                                'SELL': float(pred_proba[1]),
                                'HOLD': float(pred_proba[2])
                            }
                        elif len(pred_proba) == 2:  # Binaire
                            predictions[model_name] = {
                                'BUY': float(pred_proba[1]),  # Classe 1 = BUY
                                'SELL': float(pred_proba[0]),  # Classe 0 = SELL
                                'HOLD': 0.0
                            }
            except Exception as e:
                print(f"Error in model {model_name} prediction: {e}")
                predictions[model_name] = {'BUY': 0.33, 'SELL': 0.33, 'HOLD': 0.34}
        
        # Moyenne des prédictions si plusieurs modèles
        if predictions:
            avg_pred = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
            for model_pred in predictions.values():
                for action in ['BUY', 'SELL', 'HOLD']:
                    avg_pred[action] += model_pred.get(action, 0)
            
            for action in avg_pred:
                avg_pred[action] /= len(predictions)
            
            predictions['ensemble'] = avg_pred
        
        return predictions
    
    def prepare_feature_vector(self, features: Dict) -> List[float]:
        """Prépare un vecteur de features pour les modèles"""
        
        # Features importantes pour la décision
        important_features = [
            'quantum_confidence',
            'volatility',
            'entropy',
            'momentum_strength',
            'momentum_confidence',
            'reversal_probability',
            'volatility_forecast',
            'anomaly_score',
            'opportunity_score',
            'pattern_score',
            'social_confidence',
            'current_spread',
            'wavelet_energy',
            'hilbert_coherence',
            'hurst_exponent',
            'minute_of_hour',
            'second_of_minute',
            'scalps_this_minute'
        ]
        
        vector = []
        for feature in important_features:
            value = features.get(feature, 0)
            
            # Normalisation basique
            if feature in ['minute_of_hour', 'second_of_minute']:
                normalized = value / 60.0
            elif feature == 'scalps_this_minute':
                normalized = value / self.config.scalp_per_minute_target
            elif feature in ['current_spread']:
                normalized = min(value / 10.0, 1.0)  # Normalisé à 10 pips
            else:
                normalized = min(max(value, 0.0), 1.0)
            
            vector.append(normalized)
        
        # Padding si nécessaire
        target_length = 50
        if len(vector) < target_length:
            vector.extend([0.0] * (target_length - len(vector)))
        elif len(vector) > target_length:
            vector = vector[:target_length]
        
        return vector
    
    def synthesize_decisions(self, predictions: Dict, features: Dict, current_time: datetime) -> Dict:
        """Synthétise les décisions de tous les modèles"""
        
        # Obtenir la prédiction d'ensemble
        ensemble_pred = predictions.get('ensemble', {'BUY': 0.33, 'SELL': 0.33, 'HOLD': 0.34})
        
        # Déterminer l'action avec la plus haute probabilité
        best_action = max(ensemble_pred, key=ensemble_pred.get)
        best_confidence = ensemble_pred[best_action]
        
        # Facteurs contextuels
        context_factors = self.evaluate_context_factors(features, current_time)
        
        # Ajustement de la confiance
        adjusted_confidence = best_confidence * context_factors.get('context_score', 1.0)
        
        # Seuil de décision
        action_threshold = 0.55  # 55% de confiance minimum
        
        if adjusted_confidence > action_threshold and best_action != 'HOLD':
            # Calcul du score de risque
            risk_score = self.calculate_risk_score(features, best_action)
            
            # Raison de la décision
            reason = self.generate_decision_reason(best_action, features, predictions, context_factors)
            
            return {
                'action': best_action,
                'confidence': adjusted_confidence,
                'risk_score': risk_score,
                'reason': reason,
                'context_factors': context_factors
            }
        else:
            return {
                'action': 'HOLD',
                'confidence': adjusted_confidence,
                'risk_score': 0.5,
                'reason': 'Confidence below threshold or HOLD recommended',
                'context_factors': context_factors
            }
    
    def evaluate_context_factors(self, features: Dict, current_time: datetime) -> Dict:
        """Évalue les facteurs contextuels"""
        
        factors = {
            'time_score': 1.0,
            'market_condition_score': 1.0,
            'scalp_timing_score': 1.0,
            'context_score': 1.0
        }
        
        # Facteur temps
        minute = current_time.minute
        second = current_time.second
        
        # Meilleurs moments pour scalper (empirique)
        good_scalp_times = [3, 8, 15, 25, 40, 55]  # Secondes dans la minute
        
        if second in good_scalp_times:
            factors['time_score'] = 1.2  # Bonus de 20%
        elif minute % 5 == 0:  # Début de bougie de 5 minutes
            factors['time_score'] = 1.1
        else:
            factors['time_score'] = 0.9  # Légère pénalité
        
        # Conditions de marché
        volatility = features.get('volatility', 0)
        spread = features.get('current_spread', 0)
        
        if 0.1 < volatility < 0.5 and spread < 5.0:  # Volatilité modérée, spread serré
            factors['market_condition_score'] = 1.2
        elif volatility > 0.7 or spread > 10.0:  # Trop volatil ou spread trop large
            factors['market_condition_score'] = 0.6
        else:
            factors['market_condition_score'] = 1.0
        
        # Timing de scalp
        scalps_this_minute = features.get('scalps_this_minute', 0)
        if scalps_this_minute == 0:
            factors['scalp_timing_score'] = 1.2  # Premier scalp de la minute
        elif scalps_this_minute < self.config.scalp_per_minute_target:
            factors['scalp_timing_score'] = 1.0
        else:
            factors['scalp_timing_score'] = 0.5  # Trop de scalps
        
        # Score contextuel global
        factors['context_score'] = (
            factors['time_score'] * 0.3 +
            factors['market_condition_score'] * 0.4 +
            factors['scalp_timing_score'] * 0.3
        )
        
        return factors
    
    def calculate_risk_score(self, features: Dict, action: str) -> float:
        """Calcule le score de risque pour l'action"""
        
        risk_factors = []
        
        # 1. Volatilité
        volatility = features.get('volatility', 0)
        if volatility > 0.6:
            risk_factors.append(0.8)
        elif volatility > 0.4:
            risk_factors.append(0.6)
        elif volatility > 0.2:
            risk_factors.append(0.4)
        else:
            risk_factors.append(0.2)
        
        # 2. État quantique
        quantum_state = features.get('quantum_state', '')
        if 'DECOHERENCE' in quantum_state or 'REVOLUTION' in quantum_state:
            risk_factors.append(0.9)
        elif 'COHERENCE' in quantum_state:
            risk_factors.append(0.3)
        else:
            risk_factors.append(0.5)
        
        # 3. Momentum
        momentum_strength = features.get('momentum_strength', 0)
        if momentum_strength > 0.7:
            risk_factors.append(0.3)  # Fort momentum = risque réduit
        elif momentum_strength < 0.3:
            risk_factors.append(0.7)  # Faible momentum = risque accru
        else:
            risk_factors.append(0.5)
        
        # 4. Probabilité de renversement
        reversal_prob = features.get('reversal_probability', 0)
        if reversal_prob > 0.6:
            risk_factors.append(0.8)
        elif reversal_prob > 0.4:
            risk_factors.append(0.6)
        else:
            risk_factors.append(0.4)
        
        # 5. Spread
        spread = features.get('current_spread', 0)
        if spread > 8.0:
            risk_factors.append(0.9)
        elif spread > 5.0:
            risk_factors.append(0.7)
        elif spread > 3.0:
            risk_factors.append(0.5)
        else:
            risk_factors.append(0.3)
        
        # Score de risque moyen
        if risk_factors:
            risk_score = np.mean(risk_factors)
        else:
            risk_score = 0.5
        
        return float(risk_score)
    
    def generate_decision_reason(self, action: str, features: Dict, predictions: Dict, context_factors: Dict) -> str:
        """Génère une raison explicative pour la décision"""
        
        reasons = []
        
        # Raison principale basée sur la prédiction
        if action == 'BUY':
            reasons.append("Ensemble models suggest BUY")
        elif action == 'SELL':
            reasons.append("Ensemble models suggest SELL")
        
        # Facteurs de support
        quantum_state = features.get('quantum_state', '')
        if 'COHERENCE' in quantum_state:
            reasons.append("Quantum coherence detected")
        elif 'REVOLUTION' in quantum_state:
            reasons.append("Market revolution in progress")
        
        momentum_direction = features.get('momentum_direction', 'NEUTRAL')
        if (action == 'BUY' and momentum_direction == 'BULLISH') or \
           (action == 'SELL' and momentum_direction == 'BEARISH'):
            reasons.append(f"Momentum confirms {momentum_direction.lower()} direction")
        
        social_class = features.get('social_class', '')
        if 'RICHE' in social_class and action == 'SELL':
            reasons.append("Targeting rich class for decay")
        elif 'PAUVRE' in social_class and action == 'BUY':
            reasons.append("Targeting poor class for revolution")
        
        # Facteurs de timing
        if context_factors.get('time_score', 1.0) > 1.1:
            reasons.append("Optimal scalp timing")
        
        # Score d'opportunité
        opportunity_score = features.get('opportunity_score', 0)
        if opportunity_score > 0.6:
            reasons.append(f"High opportunity score ({opportunity_score:.1%})")
        
        # Combiner les raisons
        if reasons:
            return " | ".join(reasons[:3])  # Maximum 3 raisons
        else:
            return "Algorithmic decision based on multiple factors"
    
    def is_scalp_opportunity(self, current_time: datetime, decision: Dict) -> bool:
        """Détermine si c'est un bon moment pour scalper"""
        
        # Vérifier l'action
        if decision['action'] == 'HOLD':
            return False
        
        # Vérifier la confiance
        if decision['confidence'] < 0.55:
            return False
        
        # Vérifier le risque
        if decision.get('risk_score', 0.5) > 0.7:
            return False
        
        # Vérifier le timing dans la minute
        second = current_time.second
        scalp_times = self.config.scalp_opportunities
        
        # Priorité aux moments de scalp configurés
        if second in scalp_times:
            return True
        
        # Autres moments si conditions exceptionnelles
        if decision['confidence'] > 0.75 and decision.get('risk_score', 0.5) < 0.4:
            return True
        
        return False
    
    def calculate_lot_size(self, confidence: float, features: Dict) -> float:
        """Calcule la taille du lot"""
        
        # Taille de base
        base_lot = self.config.lot_base
        
        # Multiplicateur de confiance
        confidence_multiplier = 0.5 + confidence * 0.5  # 0.5 à 1.0
        
        # Multiplicateur de volatilité
        volatility = features.get('volatility', 0.3)
        volatility_multiplier = 1.0 / (1.0 + volatility)  # Inverse à la volatilité
        
        # Multiplicateur de risque
        risk_score = features.get('risk_score', 0.5)
        risk_multiplier = 1.5 - risk_score  # 1.0 à 0.5 (plus de risque = lot plus petit)
        
        # Calcul final
        lot_size = base_lot * confidence_multiplier * volatility_multiplier * risk_multiplier
        
        # Limites
        lot_size = max(lot_size, base_lot * 0.1)  # Minimum 10% du lot de base
        lot_size = min(lot_size, self.config.max_lot_size)  # Maximum configuré
        
        # Arrondi
        return round(lot_size, 2)
    
    def calculate_take_profit(self, features: Dict) -> float:
        """Calcule le take profit en pips"""
        
        base_tp = self.config.profit_target_initial
        
        # Ajustement par volatilité
        volatility = features.get('volatility', 0.3)
        if volatility > 0.5:
            tp_multiplier = 1.5
        elif volatility > 0.3:
            tp_multiplier = 1.2
        elif volatility < 0.1:
            tp_multiplier = 0.7
        else:
            tp_multiplier = 1.0
        
        # Ajustement par confiance
        confidence = features.get('quantum_confidence', 0.5)
        confidence_multiplier = 0.8 + confidence * 0.4  # 0.8 à 1.2
        
        tp = base_tp * tp_multiplier * confidence_multiplier
        
        return float(tp)
    
    def calculate_stop_loss(self, features: Dict) -> float:
        """Calcule le stop loss en pips"""
        
        base_sl = abs(self.config.stop_loss_initial)
        
        # Ajustement par volatilité
        volatility = features.get('volatility', 0.3)
        if volatility > 0.5:
            sl_multiplier = 1.8
        elif volatility > 0.3:
            sl_multiplier = 1.5
        elif volatility < 0.1:
            sl_multiplier = 1.2
        else:
            sl_multiplier = 1.5  # Par défaut plus large que TP
        
        # Ajustement par risque
        risk_score = features.get('risk_score', 0.5)
        risk_multiplier = 0.8 + risk_score * 0.4  # 0.8 à 1.2
        
        sl = base_sl * sl_multiplier * risk_multiplier
        
        return float(-sl)  # Négatif car c'est une perte
    
    def calculate_scalp_duration(self, features: Dict) -> int:
        """Calcule la durée maximale du scalp"""
        
        base_duration = self.config.scalp_duration_max
        
        # Ajustement par volatilité
        volatility = features.get('volatility', 0.3)
        if volatility > 0.5:
            duration_multiplier = 0.7  # Durée réduite en haute volatilité
        elif volatility < 0.2:
            duration_multiplier = 1.2  # Durée augmentée en basse volatilité
        else:
            duration_multiplier = 1.0
        
        duration = int(base_duration * duration_multiplier)
        
        # Limites
        duration = max(duration, 15)  # Minimum 15 secondes
        duration = min(duration, 90)  # Maximum 90 secondes
        
        return duration
    
    def learn_from_trade(self, trade: QuantumTrade, outcome: str):
        """Apprend d'un trade complété"""
        
        # Ajouter à la mémoire d'apprentissage
        self.learning_memory.add_experience(trade, {})
        
        # Mettre à jour les taux de succès
        if outcome == 'WIN':
            self.scalp_success_rates.append(1.0)
        else:
            self.scalp_success_rates.append(0.0)
        
        # Mettre à jour le tracker de performance
        self.performance_tracker.update(trade)
        
        # Réentraîner périodiquement
        if len(self.learning_memory.experiences) % self.config.replay_batch_size == 0:
            self.retrain_models()

# ============================================================================
# SYSTÈME D'APPRENTISSAGE ET D'ÉVOLUTION
# ============================================================================

class PerformanceTracker:
    """Tracker de performance détaillé"""
    
    def __init__(self):
        self.trades = []
        self.wins = 0
        self.losses = 0
        self.break_even = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.equity = 0.0
        
        # Métriques avancées
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.profit_factor = 0.0
        self.recovery_factor = 0.0
        self.expectancy = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.win_rate = 0.0
        
        # Par classe sociale
        self.performance_by_class = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'losses': 0, 'profit': 0.0
        })
        
        # Par état quantique
        self.performance_by_quantum = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'losses': 0, 'profit': 0.0
        })
    
    def update(self, trade: QuantumTrade):
        """Met à jour les métriques avec un nouveau trade"""
        
        self.trades.append(trade)
        
        if trade.profit is not None:
            self.equity += trade.profit
            
            if trade.profit > 0:
                self.wins += 1
                self.total_profit += trade.profit
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
                
                # Mettre à jour avg_win
                if self.wins > 0:
                    self.avg_win = self.total_profit / self.wins
            
            elif trade.profit < 0:
                self.losses += 1
                self.total_loss += abs(trade.profit)
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
                
                # Mettre à jour avg_loss
                if self.losses > 0:
                    self.avg_loss = self.total_loss / self.losses
            
            else:
                self.break_even += 1
            
            # Mettre à jour le drawdown
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = self.peak_equity - self.equity
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Mettre à jour les métriques dérivées
            self.calculate_derived_metrics()
            
            # Mettre à jour les métriques par catégorie
            self.update_category_metrics(trade)
    
    def calculate_derived_metrics(self):
        """Calcule les métriques dérivées"""
        
        total_trades = self.wins + self.losses + self.break_even
        
        if total_trades > 0:
            self.win_rate = self.wins / total_trades
        
        if self.losses > 0 and self.total_profit > 0:
            self.profit_factor = self.total_profit / self.total_loss
        
        if self.wins > 0 and self.losses > 0:
            self.expectancy = (self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)
        
        # Sharpe et Sortino (simplifiés)
        if total_trades > 1:
            profits = [t.profit for t in self.trades if t.profit is not None]
            if profits:
                avg_return = np.mean(profits)
                std_return = np.std(profits)
                
                if std_return > 0:
                    self.sharpe_ratio = avg_return / std_return
                
                # Sortino (seulement déviations négatives)
                negative_returns = [r for r in profits if r < 0]
                if negative_returns:
                    downside_std = np.std(negative_returns)
                    if downside_std > 0:
                        self.sortino_ratio = avg_return / downside_std
    
    def update_category_metrics(self, trade: QuantumTrade):
        """Met à jour les métriques par catégorie"""
        
        # Par classe sociale
        social_class = trade.social_class_targeted
        if social_class:
            class_metrics = self.performance_by_class[social_class]
            class_metrics['trades'] += 1
            
            if trade.profit is not None:
                if trade.profit > 0:
                    class_metrics['wins'] += 1
                else:
                    class_metrics['losses'] += 1
                
                class_metrics['profit'] += trade.profit
        
        # Par état quantique
        quantum_state = trade.quantum_state_at_entry
        if quantum_state:
            quantum_metrics = self.performance_by_quantum[quantum_state]
            quantum_metrics['trades'] += 1
            
            if trade.profit is not None:
                if trade.profit > 0:
                    quantum_metrics['wins'] += 1
                else:
                    quantum_metrics['losses'] += 1
                
                quantum_metrics['profit'] += trade.profit
    
    def get_summary(self) -> Dict:
        """Retourne un résumé des performances"""
        
        total_trades = self.wins + self.losses + self.break_even
        
        return {
            'total_trades': total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'break_even': self.break_even,
            'win_rate': self.win_rate,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'net_profit': self.total_profit - self.total_loss,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'avg_trade': (self.total_profit - self.total_loss) / total_trades if total_trades > 0 else 0,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'equity': self.equity,
            'peak_equity': self.peak_equity,
            'performance_by_class': dict(self.performance_by_class),
            'performance_by_quantum': dict(self.performance_by_quantum),
            'best_class': max(self.performance_by_class.items(), 
                            key=lambda x: x[1].get('profit', 0))[0] if self.performance_by_class else 'NONE',
            'best_quantum_state': max(self.performance_by_quantum.items(), 
                                    key=lambda x: x[1].get('profit', 0))[0] if self.performance_by_quantum else 'NONE'
        }

class ParameterOptimizer:
    """Optimiseur de paramètres dynamique"""
    
    def __init__(self):
        self.optimization_history = []
        self.current_params = {}
        self.best_params = {}
        self.best_score = -float('inf')
        
    def optimize(self, performance_data: Dict, current_params: Dict) -> Dict:
        """Optimise les paramètres basé sur la performance"""
        
        # Scores de performance
        win_rate = performance_data.get('win_rate', 0)
        profit_factor = performance_data.get('profit_factor', 0)
        expectancy = performance_data.get('expectancy', 0)
        sharpe_ratio = performance_data.get('sharpe_ratio', 0)
        
        # Score composite
        composite_score = (
            win_rate * 0.25 +
            min(profit_factor, 5) * 0.25 +  # Cap profit factor à 5
            expectancy * 10 * 0.25 +        # Scale expectancy
            sharpe_ratio * 0.25
        )
        
        # Enregistrer dans l'historique
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'params': current_params,
            'score': composite_score,
            'performance': performance_data
        })
        
        # Vérifier si c'est le meilleur score
        if composite_score > self.best_score:
            self.best_score = composite_score
            self.best_params = current_params.copy()
        
        # Générer de nouveaux paramètres
        new_params = self.generate_new_params(current_params, composite_score)
        
        return new_params
    
    def generate_new_params(self, current_params: Dict, current_score: float) -> Dict:
        """Génère de nouveaux paramètres"""
        
        new_params = current_params.copy()
        
        # Si le score est bon, faire de petits ajustements
        if current_score > 0.5:
            # Exploration locale
            for key in new_params:
                if isinstance(new_params[key], (int, float)):
                    # Petit ajustement aléatoire
                    adjustment = np.random.normal(0, 0.05)  # 5% d'ajustement
                    new_params[key] *= (1 + adjustment)
        else:
            # Si le score est mauvais, exploration plus agressive
            for key in new_params:
                if isinstance(new_params[key], (int, float)):
                    # Ajustement plus important
                    adjustment = np.random.normal(0, 0.15)  # 15% d'ajustement
                    new_params[key] *= (1 + adjustment)
        
        # Garder les paramètres dans des plages raisonnables
        new_params = self.clamp_params(new_params)
        
        return new_params
    
    def clamp_params(self, params: Dict) -> Dict:
        """Garde les paramètres dans des plages acceptables"""
        
        param_ranges = {
            'scalp_per_minute_target': (1, 8),
            'risk_per_trade_base': (0.1, 2.0),
            'profit_target_initial': (0.5, 10.0),
            'stop_loss_initial': (-1.0, -10.0),
            'scalp_duration_max': (30, 120),
            'lot_growth_factor': (1.0, 1.5)
        }
        
        for key, (min_val, max_val) in param_ranges.items():
            if key in params:
                params[key] = max(min_val, min(max_val, params[key]))
        
        return params
    
    def get_optimization_summary(self) -> Dict:
        """Retourne un résumé de l'optimisation"""
        
        if not self.optimization_history:
            return {'status': 'NO_OPTIMIZATION_YET'}
        
        latest = self.optimization_history[-1]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'best_score': self.best_score,
            'best_params': self.best_params,
            'current_score': latest['score'],
            'improvement': self.best_score - latest['score'],
            'optimization_trend': self.calculate_trend(),
            'recommended_adjustments': self.recommend_adjustments()
        }
    
    def calculate_trend(self) -> str:
        """Calcule la tendance d'optimisation"""
        
        if len(self.optimization_history) < 10:
            return 'INSUFFICIENT_DATA'
        
        recent_scores = [h['score'] for h in self.optimization_history[-10:]]
        
        if len(recent_scores) >= 2:
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            if trend > 0.01:
                return 'IMPROVING'
            elif trend < -0.01:
                return 'DECLINING'
            else:
                return 'STABLE'
        
        return 'UNKNOWN'
    
    def recommend_adjustments(self) -> List[str]:
        """Recommande des ajustements basés sur l'optimisation"""
        
        recommendations = []
        
        if not self.best_params:
            return recommendations
        
        # Comparer les paramètres actuels aux meilleurs
        for key, best_value in self.best_params.items():
            if key in self.current_params:
                current_value = self.current_params[key]
                diff = (current_value - best_value) / best_value
                
                if abs(diff) > 0.1:  # Plus de 10% de différence
                    if diff > 0:
                        recommendations.append(f"Reduce {key} by {abs(diff*100):.1f}%")
                    else:
                        recommendations.append(f"Increase {key} by {abs(diff*100):.1f}%")
        
        return recommendations[:3]  # Retourner les 3 recommandations principales

class StrategyEvolver:
    """Système d'évolution de stratégie"""
    
    def __init__(self):
        self.strategy_pool = []
        self.generation = 0
        self.best_strategies = []
        
    def evolve(self, performance_data: Dict, current_strategy: Dict) -> Dict:
        """Fait évoluer la stratégie"""
        
        # Évaluer la stratégie actuelle
        current_fitness = self.calculate_fitness(performance_data)
        
        # Ajouter à la piscine
        strategy_entry = {
            'strategy': current_strategy,
            'fitness': current_fitness,
            'generation': self.generation,
            'timestamp': datetime.now()
        }
        
        self.strategy_pool.append(strategy_entry)
        
        # Garder seulement les meilleures stratégies
        self.strategy_pool.sort(key=lambda x: x['fitness'], reverse=True)
        self.strategy_pool = self.strategy_pool[:10]  # Garder les 10 meilleures
        
        # Sélectionner les parents
        parents = self.select_parents()
        
        # Croisement et mutation
        new_strategy = self.crossover_and_mutate(parents, current_strategy)
        
        self.generation += 1
        
        return new_strategy
    
    def calculate_fitness(self, performance_data: Dict) -> float:
        """Calcule le fitness d'une stratégie"""
        
        win_rate = performance_data.get('win_rate', 0)
        profit_factor = performance_data.get('profit_factor', 0)
        sharpe_ratio = performance_data.get('sharpe_ratio', 0)
        max_drawdown = performance_data.get('max_drawdown', 1.0)
        
        # Score de fitness composite
        fitness = (
            win_rate * 0.3 +
            min(profit_factor, 3) * 0.3 +  # Cap à 3
            sharpe_ratio * 0.2 +
            (1.0 / (1.0 + max_drawdown)) * 0.2  # Inverse au drawdown
        )
        
        return fitness
    
    def select_parents(self) -> List[Dict]:
        """Sélectionne les parents pour la reproduction"""
        
        if len(self.strategy_pool) < 2:
            return self.strategy_pool
        
        # Sélection par roulette biaisée par le fitness
        total_fitness = sum(s['fitness'] for s in self.strategy_pool)
        
        if total_fitness > 0:
            probabilities = [s['fitness'] / total_fitness for s in self.strategy_pool]
            parents_idx = np.random.choice(len(self.strategy_pool), 
                                          size=min(2, len(self.strategy_pool)),
                                          p=probabilities,
                                          replace=False)
            
            return [self.strategy_pool[i] for i in parents_idx]
        
        return self.strategy_pool[:2]
    
    def crossover_and_mutate(self, parents: List[Dict], current_strategy: Dict) -> Dict:
        """Effectue le croisement et la mutation"""
        
        if len(parents) < 2:
            return current_strategy
        
        # Croisement
        parent1 = parents[0]['strategy']
        parent2 = parents[1]['strategy']
        
        new_strategy = {}
        
        for key in current_strategy.keys():
            if key in parent1 and key in parent2:
                # Croisement aléatoire
                if np.random.random() > 0.5:
                    new_strategy[key] = parent1[key]
                else:
                    new_strategy[key] = parent2[key]
            
            # Mutation
            if isinstance(new_strategy.get(key), (int, float)):
                if np.random.random() < 0.1:  # 10% de chance de mutation
                    mutation = np.random.normal(0, 0.1)  # 10% de mutation
                    new_strategy[key] *= (1 + mutation)
        
        # S'assurer que tous les champs nécessaires sont présents
        for key in current_strategy:
            if key not in new_strategy:
                new_strategy[key] = current_strategy[key]
        
        return new_strategy
    
    def get_evolution_summary(self) -> Dict:
        """Retourne un résumé de l'évolution"""
        
        if not self.strategy_pool:
            return {'status': 'NO_EVOLUTION_YET'}
        
        best_strategy = self.strategy_pool[0]
        
        return {
            'generation': self.generation,
            'best_fitness': best_strategy['fitness'],
            'best_strategy_age': (datetime.now() - best_strategy['timestamp']).total_seconds(),
            'strategy_pool_size': len(self.strategy_pool),
            'fitness_range': {
                'min': min(s['fitness'] for s in self.strategy_pool),
                'max': best_strategy['fitness'],
                'avg': np.mean([s['fitness'] for s in self.strategy_pool])
            },
            'evolution_progress': self.calculate_progress()
        }
    
    def calculate_progress(self) -> str:
        """Calcule le progrès de l'évolution"""
        
        if len(self.strategy_pool) < 5:
            return 'EARLY_STAGE'
        
        # Vérifier l'amélioration sur les dernières générations
        recent_fitness = [s['fitness'] for s in self.strategy_pool[-5:]]
        
        if len(recent_fitness) >= 2:
            improvement = recent_fitness[-1] - recent_fitness[0]
            
            if improvement > 0.05:
                return 'SIGNIFICANT_IMPROVEMENT'
            elif improvement > 0.01:
                return 'SLOW_IMPROVEMENT'
            elif improvement < -0.05:
                return 'DECLINING'
            else:
                return 'STAGNANT'
        
        return 'UNKNOWN'

# ============================================================================
# EXÉCUTEUR DE TRADES AVANCÉ
# ============================================================================

class QuantumTradeExecutor:
    """Exécuteur de trades quantique avancé"""
    
    def __init__(self, config: QuantumConfig, decision_system: QuantumDecisionSystem):
        self.config = config
        self.decision_system = decision_system
        self.mt5_connected = False
        self.open_trades = {}
        self.trade_history = []
        self.trade_counter = 0
        
        # Connexion MT5
        self.connect_mt5()
    
    def connect_mt5(self) -> bool:
        """Connecte à MT5"""
        
        print("\n🔌 Connexion à MetaTrader 5...")
        
        if not mt5.initialize():
            print("❌ Échec de l'initialisation MT5")
            return False
        
        authorized = mt5.login(
            login=self.config.login,
            password=self.config.password,
            server=self.config.server
        )
        
        if not authorized:
            print("❌ Échec de l'authentification MT5")
            mt5.shutdown()
            return False
        
        account_info = mt5.account_info()
        if account_info:
            print(f"✅ Connecté au compte #{account_info.login}")
            print(f"   Équité: ${account_info.equity:,.2f}")
            print(f"   Marge libre: ${account_info.margin_free:,.2f}")
            print(f"   Balance: ${account_info.balance:,.2f}")
        
        self.mt5_connected = True
        return True
    
    def execute_trade(self, decision: Dict) -> Optional[QuantumTrade]:
        """Exécute un trade basé sur la décision"""
        
        if not self.mt5_connected:
            print("❌ MT5 non connecté")
            return None
        
        if decision.get('action') == 'HOLD':
            return None
        
        # Préparer la requête de trade
        symbol_info = mt5.symbol_info(self.config.symbol)
        if not symbol_info:
            print(f"❌ Symbole {self.config.symbol} non trouvé")
            return None
        
        # Calculer les prix
        tick = mt5.symbol_info_tick(self.config.symbol)
        if not tick:
            print("❌ Impossible d'obtenir le tick actuel")
            return None
        
        if decision['action'] == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            stop_loss = price - (decision.get('stop_loss', 4.0) / 10000)  # Convertir pips en prix
            take_profit = price + (decision.get('take_profit', 2.0) / 10000)
        else:  # SELL
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            stop_loss = price + (abs(decision.get('stop_loss', -4.0)) / 10000)
            take_profit = price - (decision.get('take_profit', 2.0) / 10000)
        
        # Préparer la requête
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": decision.get('lot_size', self.config.lot_base),
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 888888,
            "comment": f"QUANTUM_{decision.get('social_class', 'UNKNOWN')}_{decision.get('quantum_state', 'UNKNOWN')}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Envoyer l'ordre
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            # Créer l'objet QuantumTrade
            trade = QuantumTrade(
                trade_id=f"QT_{self.trade_counter:06d}",
                open_time=datetime.now(),
                direction=decision['action'],
                entry_price=price,
                lot_size=decision.get('lot_size', self.config.lot_base),
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantum_state_at_entry=decision.get('quantum_state', 'UNKNOWN'),
                social_class_targeted=decision.get('social_class', 'UNKNOWN'),
                genome_id=decision.get('genome_id', 'UNKNOWN'),
                decision_confidence=decision.get('confidence', 0.0),
                decision_reason=decision.get('reason', ''),
                decision_algorithm='QUANTUM_ENSEMBLE',
                risk_score=decision.get('risk_score', 0.5),
                scalp_number=decision.get('scalp_number', 0),
                minute_of_day=datetime.now().minute,
                market_regime=decision.get('market_regime', 'UNKNOWN'),
                learning_features=decision.get('features', {})
            )
            
            self.trade_counter += 1
            self.open_trades[result.order] = trade
            
            print(f"\n🎯 TRADE EXÉCUTÉ #{result.order}!")
            print(f"   Action: {decision['action']}")
            print(f"   Prix: ${price:.2f}")
            print(f"   Lot: {decision.get('lot_size', 0.01):.2f}")
            print(f"   SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")
            print(f"   Raison: {decision.get('reason', '')[:50]}...")
            print(f"   Confiance: {decision.get('confidence', 0):.1%}")
            
            return trade
        
        else:
            print(f"❌ Échec de l'exécution du trade: {result.comment if result else 'Unknown error'}")
            return None
    
    def manage_open_trades(self):
        """Gère les trades ouverts"""
        
        if not self.open_trades:
            return
        
        positions = mt5.positions_get(symbol=self.config.symbol, magic=888888)
        current_time = datetime.now()
        
        for position in positions:
            if position.ticket in self.open_trades:
                trade = self.open_trades[position.ticket]
                
                # Mettre à jour le profit en temps réel
                trade.calculate_results(position.price_current if hasattr(position, 'price_current') else None)
                
                # Vérifier les conditions de sortie
                exit_reason = self.check_exit_conditions(trade, position, current_time)
                
                if exit_reason:
                    self.close_trade(position, trade, exit_reason)
    
    def check_exit_conditions(self, trade: QuantumTrade, position, current_time: datetime) -> Optional[str]:
        """Vérifie les conditions de sortie pour un trade"""
        
        # 1. Take Profit atteint
        if trade.profit is not None and trade.profit >= trade.take_profit * trade.lot_size * 100000:
            return "TAKE_PROFIT"
        
        # 2. Stop Loss atteint
        if trade.profit is not None and trade.profit <= trade.stop_loss * trade.lot_size * 100000:
            return "STOP_LOSS"
        
        # 3. Trailing Stop (si activé)
        if trade.trailing_stop_activated and trade.trailing_stop_price:
            if (trade.direction == 'BUY' and position.price_current <= trade.trailing_stop_price) or \
               (trade.direction == 'SELL' and position.price_current >= trade.trailing_stop_price):
                return "TRAILING_STOP"
        
        # 4. Timeout
        trade_duration = (current_time - trade.open_time).total_seconds()
        scalp_duration = trade.learning_features.get('scalp_duration', self.config.scalp_duration_max)
        
        if trade_duration > scalp_duration:
            return "TIMEOUT"
        
        # 5. Changement d'état quantique (à implémenter avec des données en temps réel)
        # 6. Signal de sortie du système de décision (à implémenter)
        
        return None
    
    def close_trade(self, position, trade: QuantumTrade, reason: str):
        """Ferme un trade"""
        
        # Déterminer le type d'ordre de fermeture
        if position.type == 0:  # BUY position
            close_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(self.config.symbol).bid
        else:  # SELL position
            close_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.config.symbol).ask
        
        # Préparer la requête de fermeture
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": 888888,
            "comment": f"CLOSE_{reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Envoyer l'ordre de fermeture
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            # Mettre à jour le trade
            trade.close_time = datetime.now()
            trade.exit_price = price
            trade.quantum_state_at_exit = 'COLLAPSE'  # À raffiner avec l'état actuel
            trade.calculate_results()
            
            # Déterminer l'issue
            if trade.profit is not None:
                if trade.profit > 0:
                    trade.outcome_classification = 'WIN'
                    trade.lesson_learned = f"Successful {reason.lower()}"
                elif trade.profit < 0:
                    trade.outcome_classification = 'LOSS'
                    trade.lesson_learned = f"Stopped by {reason.lower()}"
                else:
                    trade.outcome_classification = 'BREAK_EVEN'
                    trade.lesson_learned = "Break even exit"
            
            # Ajouter à l'historique
            self.trade_history.append(trade)
            
            # Retirer des trades ouverts
            if position.ticket in self.open_trades:
                del self.open_trades[position.ticket]
            
            # Apprentissage
            self.decision_system.learn_from_trade(trade, trade.outcome_classification)
            
            print(f"✅ TRADE FERMÉ #{position.ticket}: ${trade.profit:+.2f} ({reason})")
            
            return True
        
        return False
    
    def get_trade_statistics(self) -> Dict:
        """Retourne les statistiques des trades"""
        
        if not self.trade_history:
            return {'total_trades': 0, 'message': 'No trades yet'}
        
        wins = sum(1 for t in self.trade_history if t.outcome_classification == 'WIN')
        losses = sum(1 for t in self.trade_history if t.outcome_classification == 'LOSS')
        break_even = sum(1 for t in self.trade_history if t.outcome_classification == 'BREAK_EVEN')
        
        total_profit = sum(t.profit for t in self.trade_history if t.profit is not None and t.profit > 0)
        total_loss = sum(abs(t.profit) for t in self.trade_history if t.profit is not None and t.profit < 0)
        
        win_rate = wins / len(self.trade_history) if self.trade_history else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            'total_trades': len(self.trade_history),
            'wins': wins,
            'losses': losses,
            'break_even': break_even,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': total_profit - total_loss,
            'profit_factor': profit_factor if profit_factor != float('inf') else 'INFINITE',
            'open_trades': len(self.open_trades),
            'avg_trade_duration': np.mean([t.duration_seconds for t in self.trade_history if t.duration_seconds]) if any(t.duration_seconds for t in self.trade_history) else 0
        }

# ============================================================================
# SYSTÈME PRINCIPAL DE TRADING QUANTIQUE
# ============================================================================

class QuantumTradingSystem:
    """SYSTÈME PRINCIPAL DE TRADING QUANTIQUE"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        
        # Initialiser les composants
        self.quantum_engine = QuantumEngine(config)
        self.decision_system = QuantumDecisionSystem(config)
        self.trade_executor = QuantumTradeExecutor(config, self.decision_system)
        
        # Données en temps réel
        self.tick_stream = deque(maxlen=10000)
        self.market_data = {}
        self.last_analysis_time = time.time()
        self.last_decision_time = time.time()
        self.last_scalp_time = time.time()
        
        # Dashboard et monitoring
        self.dashboard = QuantumDashboard()
        self.performance_monitor = PerformanceMonitor()
        
        # Threads
        self.running = False
        self.analysis_thread = None
        self.decision_thread = None
        self.management_thread = None
        self.dashboard_thread = None
    
    def start(self):
        """Démarre le système de trading"""
        
        print("\n" + "="*100)
        print("🚀 DÉMARRAGE DU SYSTÈME DE TRADING QUANTIQUE")
        print("="*100)
        
        if not self.trade_executor.mt5_connected:
            print("❌ Impossible de se connecter à MT5")
            return
        
        self.running = True
        
        # Démarrer les threads
        self.analysis_thread = threading.Thread(target=self.analysis_loop, daemon=True)
        self.decision_thread = threading.Thread(target=self.decision_loop, daemon=True)
        self.management_thread = threading.Thread(target=self.management_loop, daemon=True)
        self.dashboard_thread = threading.Thread(target=self.dashboard_loop, daemon=True)
        
        self.analysis_thread.start()
        self.decision_thread.start()
        self.management_thread.start()
        self.dashboard_thread.start()
        
        print("\n✅ Système démarré avec succès!")
        print("   Press Ctrl+C to stop\n")
        
        # Boucle principale
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n🛑 Arrêt demandé par l'utilisateur")
            self.stop()
    
    def analysis_loop(self):
        """Boucle d'analyse des données de marché"""
        
        print("📊 Démarrage de la boucle d'analyse...")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Collecter les ticks
                self.collect_ticks()
                
                # Analyser périodiquement
                if current_time - self.last_analysis_time >= self.config.analysis_frequency:
                    self.analyze_market()
                    self.last_analysis_time = current_time
                
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                print(f"❌ Erreur dans la boucle d'analyse: {e}")
                time.sleep(1)
    
    def decision_loop(self):
        """Boucle de prise de décision"""
        
        print("🤖 Démarrage de la boucle de décision...")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Prendre des décisions périodiquement
                if current_time - self.last_decision_time >= self.config.decision_frequency:
                    if self.market_data:
                        decision = self.decision_system.make_decision(
                            self.market_data, 
                            datetime.now()
                        )
                        
                        # Exécuter le trade si opportunité
                        if decision.get('scalp_opportunity', False):
                            # Vérifier le timing entre les scalps
                            if current_time - self.last_scalp_time >= self.config.scalp_frequency:
                                trade = self.trade_executor.execute_trade(decision)
                                if trade:
                                    self.last_scalp_time = current_time
                    
                    self.last_decision_time = current_time
                
                time.sleep(0.01)  # 10ms
                
            except Exception as e:
                print(f"❌ Erreur dans la boucle de décision: {e}")
                time.sleep(1)
    
    def management_loop(self):
        """Boucle de gestion des trades"""
        
        print("⚙️ Démarrage de la boucle de gestion...")
        
        while self.running:
            try:
                # Gérer les trades ouverts
                self.trade_executor.manage_open_trades()
                
                # Apprentissage périodique
                if self.config.learning_enabled:
                    if int(time.time()) % self.config.learning_interval == 0:
                        self.learn_from_experience()
                
                time.sleep(0.5)  # 500ms
                
            except Exception as e:
                print(f"❌ Erreur dans la boucle de gestion: {e}")
                time.sleep(5)
    
    def dashboard_loop(self):
        """Boucle d'affichage du dashboard"""
        
        print("📈 Démarrage de la boucle du dashboard...")
        
        while self.running:
            try:
                # Afficher le dashboard périodiquement
                if int(time.time()) % 5 == 0:  # Toutes les 5 secondes
                    self.update_dashboard()
                
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Erreur dans la boucle du dashboard: {e}")
                time.sleep(5)
    
    def collect_ticks(self):
        """Collecte les ticks de marché"""
        
        tick = mt5.symbol_info_tick(self.config.symbol)
        if tick:
            tick_data = {
                'timestamp': datetime.now(),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time_msc': tick.time_msc
            }
            
            self.tick_stream.append(tick_data)
    
    def analyze_market(self):
        """Analyse le marché en temps réel"""
        
        if len(self.tick_stream) < 100:
            return
        
        # Analyser avec le moteur quantique
        quantum_analysis = self.quantum_engine.analyze_tick_stream(list(self.tick_stream))
        
        # Mettre à jour les données de marché
        self.market_data = {
            'timestamp': datetime.now(),
            'tick_stream': list(self.tick_stream)[-500:],  # Derniers 500 ticks
            'quantum_analysis': quantum_analysis,
            'current_price': self.tick_stream[-1]['bid'] if self.tick_stream else 0,
            'current_spread': (self.tick_stream[-1]['ask'] - self.tick_stream[-1]['bid']) * 10000 if self.tick_stream else 0
        }
    
    def learn_from_experience(self):
        """Apprend de l'expérience accumulée"""
        
        # Obtenir les statistiques de performance
        stats = self.trade_executor.get_trade_statistics()
        
        # Optimiser les paramètres
        if self.decision_system.parameter_optimizer:
            new_params = self.decision_system.parameter_optimizer.optimize(
                stats,
                self.config.__dict__
            )
            
            # Mettre à jour la configuration
            for key, value in new_params.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Faire évoluer la stratégie
        if self.decision_system.strategy_evolver:
            new_strategy = self.decision_system.strategy_evolver.evolve(
                stats,
                self.config.__dict__
            )
            
            # Mettre à jour la configuration
            for key, value in new_strategy.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
    
    def update_dashboard(self):
        """Met à jour et affiche le dashboard"""
        
        # Récupérer les données actuelles
        stats = self.trade_executor.get_trade_statistics()
        performance = self.decision_system.performance_tracker.get_summary()
        
        # Mettre à jour le dashboard
        self.dashboard.update({
            'config': self.config,
            'market_data': self.market_data,
            'trade_stats': stats,
            'performance': performance,
            'open_trades': len(self.trade_executor.open_trades),
            'scalps_this_minute': self.decision_system.scalps_this_minute,
            'quantum_state': self.market_data.get('quantum_analysis', {}).get('quantum_synthesis', {}).get('quantum_state', 'UNKNOWN'),
            'current_time': datetime.now()
        })
        
        # Afficher le dashboard
        self.dashboard.display()
    
    def stop(self):
        """Arrête le système de trading"""
        
        print("\n🔴 Arrêt du système de trading...")
        
        self.running = False
        
        # Attendre que les threads se terminent
        if self.analysis_thread:
            self.analysis_thread.join(timeout=2)
        if self.decision_thread:
            self.decision_thread.join(timeout=2)
        if self.management_thread:
            self.management_thread.join(timeout=2)
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=2)
        
        # Fermer tous les trades ouverts
        print("📤 Fermeture des trades ouverts...")
        for position in mt5.positions_get(symbol=self.config.symbol, magic=888888):
            if position.ticket in self.trade_executor.open_trades:
                trade = self.trade_executor.open_trades[position.ticket]
                self.trade_executor.close_trade(position, trade, 'SHUTDOWN')
        
        # Sauvegarder les données
        self.save_data()
        
        # Déconnexion MT5
        if self.trade_executor.mt5_connected:
            mt5.shutdown()
        
        print("\n✅ Système arrêté avec succès!")
        print(f"   Trades totaux: {stats.get('total_trades', 0) if 'stats' in locals() else 0}")
        print(f"   Profit net: ${stats.get('net_profit', 0):.2f}" if 'stats' in locals() else "")
    
    def save_data(self):
        """Sauvegarde les données du système"""
        
        save_dir = self.config.data_directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Sauvegarder l'historique des trades
        trades_data = [trade.to_dict() for trade in self.trade_executor.trade_history]
        with open(os.path.join(save_dir, self.config.trade_log_file), 'w') as f:
            json.dump(trades_data, f, indent=2)
        
        # Sauvegarder les métriques de performance
        performance_data = self.decision_system.performance_tracker.get_summary()
        with open(os.path.join(save_dir, self.config.performance_file), 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        # Sauvegarder l'évolution de l'apprentissage
        learning_data = {
            'optimization': self.decision_system.parameter_optimizer.get_optimization_summary(),
            'evolution': self.decision_system.strategy_evolver.get_evolution_summary(),
            'config': self.config.__dict__
        }
        with open(os.path.join(save_dir, self.config.learning_log_file), 'w') as f:
            json.dump(learning_data, f, indent=2)
        
        print(f"\n💾 Données sauvegardées dans {save_dir}/")

# ============================================================================
# DASHBOARD QUANTIQUE
# ============================================================================

class QuantumDashboard:
    """Dashboard en temps réel du système quantique"""
    
    def __init__(self):
        self.data = {}
        self.last_update = datetime.now()
    
    def update(self, data: Dict):
        """Met à jour les données du dashboard"""
        self.data = data
        self.last_update = datetime.now()
    
    def display(self):
        """Affiche le dashboard"""
        
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" + "="*120)
        print("🎻 SYMPHONIE QUANTIQUE - DASHBOARD TEMPS RÉEL")
        print("="*120)
        
        # Section Prix et Marché
        market_data = self.data.get('market_data', {})
        if market_data:
            current_price = market_data.get('current_price', 0)
            current_spread = market_data.get('current_spread', 0)
            quantum_state = market_data.get('quantum_analysis', {}).get('quantum_synthesis', {}).get('quantum_state', 'UNKNOWN')
            
            print(f"\n💰 PRIX: ${current_price:.2f} | Spread: {current_spread:.1f} pips | État Quantique: {quantum_state}")
        
        # Section Performance
        stats = self.data.get('trade_stats', {})
        performance = self.data.get('performance', {})
        
        print(f"\n📈 PERFORMANCE:")
        print(f"   Trades: {stats.get('total_trades', 0)} | Win Rate: {stats.get('win_rate', 0):.1%}")
        print(f"   Profit Net: ${stats.get('net_profit', 0):+.2f} | Facteur Profit: {stats.get('profit_factor', 0):.2f}")
        print(f"   Expectancy: ${performance.get('expectancy', 0):+.2f} | Sharpe: {performance.get('sharpe_ratio', 0):.2f}")
        
        # Section Scalping
        config = self.data.get('config', {})
        scalps_this_minute = self.data.get('scalps_this_minute', 0)
        open_trades = self.data.get('open_trades', 0)
        
        print(f"\n⚡ SCALPING:")
        print(f"   Scalps cette minute: {scalps_this_minute}/{config.get('scalp_per_minute_target', 4)}")
        print(f"   Trades ouverts: {open_trades}/{config.get('max_concurrent_scalps', 3)}")
        print(f"   Lot base: {config.get('lot_base', 0.01):.2f} | Lot max: {config.get('max_lot_size', 0.50):.2f}")
        
        # Section Apprentissage
        print(f"\n🧠 APPRENTISSAGE:")
        print(f"   Mode: {config.get('operational_mode', 'AGGRESSIVE_LEARNING')}")
        print(f"   Phase: {config.get('learning_phase', 'ACQUISITION')}")
        print(f"   Meilleure classe: {performance.get('best_class', 'NONE')}")
        print(f"   Meilleur état quantique: {performance.get('best_quantum_state', 'NONE')}")
        
        # Section Trades Ouverts
        print(f"\n🎯 TRADES OUVERTS:")
        
        trade_executor = self.data.get('trade_executor', None)
        if trade_executor and hasattr(trade_executor, 'open_trades'):
            open_trades_data = trade_executor.open_trades
            if open_trades_data:
                for ticket, trade in list(open_trades_data.items())[:3]:  # Afficher les 3 premiers
                    if hasattr(trade, 'profit'):
                        profit = trade.profit if trade.profit is not None else 0
                        duration = (datetime.now() - trade.open_time).total_seconds() if trade.open_time else 0
                        
                        print(f"   #{ticket}: {trade.direction} @ ${trade.entry_price:.2f} | "
                              f"Profit: ${profit:+.2f} | Durée: {duration:.0f}s | "
                              f"Classe: {trade.social_class_targeted}")
            else:
                print("   Aucun trade ouvert")
        
        # Section Temps
        current_time = self.data.get('current_time', datetime.now())
        print(f"\n⏰ TEMPS: {current_time.strftime('%H:%M:%S')} | "
              f"Mise à jour: {(datetime.now() - self.last_update).total_seconds():.1f}s")
        
        print("\n" + "="*120)
        print("⚡ SYSTÈME ACTIF - SCALPAGE QUANTIQUE EN COURS...")
        print("="*120)

# ============================================================================
# MONITEUR DE PERFORMANCE
# ============================================================================

class PerformanceMonitor:
    """Moniteur de performance détaillé"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.alerts = deque(maxlen=50)
    
    def update(self, metrics: Dict):
        """Met à jour les métriques de performance"""
        
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        # Vérifier les alertes
        self.check_alerts(metrics)
    
    def check_alerts(self, metrics: Dict):
        """Vérifie les conditions d'alerte"""
        
        # Alerte de drawdown
        max_drawdown = metrics.get('max_drawdown', 0)
        if abs(max_drawdown) > 100:  # Drawdown de plus de 100$
            self.add_alert(f"⚠️ Drawdown élevé: ${max_drawdown:.2f}")
        
        # Alerte de pertes consécutives
        consecutive_losses = metrics.get('max_consecutive_losses', 0)
        if consecutive_losses >= 5:
            self.add_alert(f"⚠️ {consecutive_losses} pertes consécutives")
        
        # Alerte de win rate bas
        win_rate = metrics.get('win_rate', 0)
        if win_rate < 0.4 and metrics.get('total_trades', 0) > 20:
            self.add_alert(f"⚠️ Win rate bas: {win_rate:.1%}")
    
    def add_alert(self, message: str):
        """Ajoute une alerte"""
        
        self.alerts.append({
            'timestamp': datetime.now(),
            'message': message
        })
    
    def get_summary(self) -> Dict:
        """Retourne un résumé des performances"""
        
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]['metrics']
        
        return {
            'current_performance': latest,
            'alert_count': len(self.alerts),
            'recent_alerts': list(self.alerts)[-3:],  # 3 dernières alertes
            'performance_trend': self.calculate_trend(),
            'recommendations': self.generate_recommendations(latest)
        }
    
    def calculate_trend(self) -> str:
        """Calcule la tendance de performance"""
        
        if len(self.metrics_history) < 10:
            return 'INSUFFICIENT_DATA'
        
        # Analyser la tendance des profits nets
        net_profits = [m['metrics'].get('net_profit', 0) for m in self.metrics_history]
        
        if len(net_profits) >= 2:
            recent_trend = np.polyfit(range(len(net_profits[-5:])), net_profits[-5:], 1)[0]
            
            if recent_trend > 1.0:
                return 'STRONGLY_IMPROVING'
            elif recent_trend > 0.1:
                return 'IMPROVING'
            elif recent_trend < -1.0:
                return 'STRONGLY_DECLINING'
            elif recent_trend < -0.1:
                return 'DECLINING'
            else:
                return 'STABLE'
        
        return 'UNKNOWN'
    
    def generate_recommendations(self, metrics: Dict) -> List[str]:
        """Génère des recommandations basées sur les performances"""
        
        recommendations = []
        
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        max_drawdown = abs(metrics.get('max_drawdown', 0))
        
        if win_rate < 0.45:
            recommendations.append("Considérer réduire la taille des positions")
        
        if profit_factor < 1.2:
            recommendations.append("Revoir les critères d'entrée")
        
        if max_drawdown > 50:
            recommendations.append("Renforcer la gestion des risques")
        
        return recommendations[:3]

# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

def main():
    """POINT D'ENTRÉE DU SYSTÈME DE TRADING QUANTIQUE"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                🎻 SYMPHONIE QUANTIQUE ULTIMATE V2.0 🎻                      ║
║                                                                              ║
║  SYSTÈME DE SCALPAGE QUANTIQUE SOCIAL À APPRENTISSAGE AUTOMATIQUE            ║
║                                                                              ║
║  CAPACITÉS PRINCIPALES:                                                      ║
║    • Multi-scalping intelligent (2-6 trades/minute)                          ║
║    • Analyse quantique avancée (FFT, ondelettes, Hilbert, fractales)         ║
║    • Classification sociale des bougies (10 classes)                         ║
║    • Apprentissage profond en temps réel                                     ║
║    • Évolution dynamique des paramètres                                      ║
║    • Gestion de risque adaptative                                            ║
║    • Dashboard temps réel complet                                            ║
║                                                                              ║
║  MODES D'OPÉRATION:                                                          ║
║    • AGGRESSIVE_LEARNING: Scalpage agressif avec apprentissage               ║
║    • CONSERVATIVE: Prise de risque réduite                                   ║
║    • HYPER_AGGRESSIVE: Maximum de scalps                                     ║
║                                                                              ║
║  PHASES D'APPRENTISSAGE:                                                     ║
║    • ACQUISITION: Collecte de données et apprentissage initial               ║
║    • CONSOLIDATION: Raffinement des modèles                                  ║
║    • EXPLOITATION: Utilisation optimale des connaissances                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration initiale
    config = QuantumConfig()
    
    # Personnalisation interactive
    print("\n⚙️  CONFIGURATION DU SYSTÈME QUANTIQUE:")
    print("   (Appuyez sur Entrée pour les valeurs par défaut)\n")
    
    try:
        # Mode d'opération
        print("Modes disponibles:")
        print("   1. AGGRESSIVE_LEARNING (Recommandé)")
        print("   2. CONSERVATIVE")
        print("   3. HYPER_AGGRESSIVE")
        
        mode_choice = input("\nChoisissez le mode (1-3): ").strip()
        if mode_choice == '2':
            config.operational_mode = 'CONSERVATIVE'
            config.scalp_per_minute_target = 2
            config.risk_per_trade_base = 0.3
        elif mode_choice == '3':
            config.operational_mode = 'HYPER_AGGRESSIVE'
            config.scalp_per_minute_target = 6
            config.risk_per_trade_base = 0.8
        else:
            config.operational_mode = 'AGGRESSIVE_LEARNING'
        
        # Symbole
        symbol = input(f"\nSymbole à trader (défaut: {config.symbol}): ").strip()
        if symbol:
            config.symbol = symbol
        
        # Phase d'apprentissage
        print("\nPhases d'apprentissage:")
        print("   1. ACQUISITION (Nouveau marché)")
        print("   2. CONSOLIDATION (Expérience modérée)")
        print("   3. EXPLOITATION (Expérience avancée)")
        
        phase_choice = input("\nChoisissez la phase (1-3): ").strip()
        if phase_choice == '2':
            config.learning_phase = 'CONSOLIDATION'
        elif phase_choice == '3':
            config.learning_phase = 'EXPLOITATION'
            config.learning_enabled = False  # Désactiver l'apprentissage en phase d'exploitation
        else:
            config.learning_phase = 'ACQUISITION'
        
        # Capital de base
        capital_input = input(f"\nCapital de base en $ (défaut: {config.base_capital}): ").strip()
        if capital_input:
            try:
                config.base_capital = float(capital_input)
            except:
                print("Valeur invalide, utilisation de la valeur par défaut")
        
        print(f"\n✅ Configuration terminée:")
        print(f"   Mode: {config.operational_mode}")
        print(f"   Phase: {config.learning_phase}")
        print(f"   Symbole: {config.symbol}")
        print(f"   Capital: ${config.base_capital:,.2f}")
        print(f"   Scalps/minute cible: {config.scalp_per_minute_target}")
        print(f"   Risque/trade: {config.risk_per_trade_base}%")
        
    except Exception as e:
        print(f"❌ Erreur de configuration: {e}")
        print("Utilisation des valeurs par défaut")
    
    # Avertissement
    print("\n" + "="*80)
    print("⚠️  AVERTISSEMENT IMPORTANT")
    print("="*80)
    print("Ce système est très agressif et peut entraîner des pertes importantes.")
    print("Il est conçu pour des tests en compte démo uniquement.")
    print("Ne jamais utiliser sur un compte réel sans compréhension complète.")
    print("="*80)
    
    confirmation = input("\nAppuyez sur Entrée pour démarrer, ou 'q' pour quitter: ").strip()
    if confirmation.lower() == 'q':
        print("\nArrêt demandé.")
        return
    
    # Démarrer le système
    trading_system = QuantumTradingSystem(config)
    
    try:
        trading_system.start()
    except KeyboardInterrupt:
        print("\n\n🛑 Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n💥 ERREUR CRITIQUE: {e}")
        print(traceback.format_exc())
    finally:
        if 'trading_system' in locals():
            trading_system.stop()

if __name__ == "__main__":
    # Vérification des dépendances
    required_packages = [
        'MetaTrader5',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'xgboost',
        'lightgbm',
        'tensorflow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'MetaTrader5':
                import MetaTrader5
            elif package == 'numpy':
                import numpy
            elif package == 'pandas':
                import pandas
            elif package == 'scipy':
                import scipy
            elif package == 'scikit-learn':
                import sklearn
            elif package == 'xgboost':
                import xgboost
            elif package == 'lightgbm':
                import lightgbm
            elif package == 'tensorflow':
                import tensorflow
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Dépendances manquantes:")
        for package in missing_packages:
            print(f"   - {package}")
        
        print("\n📦 Installation requise:")
        print("   pip install MetaTrader5 numpy pandas scipy scikit-learn xgboost lightgbm tensorflow")
        
        if 'MetaTrader5' in missing_packages:
            print("\n💡 Pour MetaTrader5 sur Windows:")
            print("   pip install --upgrade MetaTrader5")
        
        exit(1)
    
    # Lancer le système
    main()