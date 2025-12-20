"""
NLP Experiments for osu! Rank Predictor

This module extracts text-based features from beatmap metadata,
experimenting with NLP techniques in a game analytics context.

Experiments include:
- Keyword matching for genre classification
- Pattern recognition with regex for difficulty extraction
- N-gram analysis on mod combinations
- Vocabulary diversity metrics
- Character-level language detection
"""

import numpy as np
from collections import Counter
import re


class NLPFeatureExtractor:
    """
    Extract experimental NLP features from osu! beatmap metadata.
    
    Uses lightweight methods by default (keyword matching, pattern recognition).
    Can optionally use sentence transformers for better embeddings.
    """
    
    def __init__(self, use_transformers=False):
        """
        Initialize the NLP feature extractor.
        
        Args:
            use_transformers: If True, use sentence-transformers for embeddings.
                             Requires 'sentence-transformers' package.
        """
        self.use_transformers = use_transformers
        self.encoder = None
        self.fitted = False
        
        # Common genre/style keywords for categorization
        self.genre_keywords = {
            'anime': ['anime', 'vocaloid', 'touhou', 'hatsune', 'miku', 'nico'],
            'electronic': ['electronic', 'edm', 'dubstep', 'dnb', 'drum', 'bass', 'techno', 'house', 'trance'],
            'rock': ['rock', 'metal', 'punk', 'guitar', 'band'],
            'pop': ['pop', 'kpop', 'jpop', 'idol'],
            'classical': ['classical', 'piano', 'orchestra', 'symphony'],
            'game': ['game', 'ost', 'soundtrack', 'video game', 'rhythm'],
        }
        
        # Difficulty indicators in beatmap names
        self.difficulty_keywords = {
            'easy': ['easy', 'beginner', 'normal', 'simple'],
            'medium': ['hard', 'hyper', 'advanced'],
            'hard': ['insane', 'expert', 'extreme', 'lunatic'],
            'expert': ['extra', 'nightmare', 'impossible', 'marathon'],
        }
        
        # Mod abbreviations (treated as "vocabulary")
        self.mod_vocab = ['HD', 'HR', 'DT', 'NC', 'FL', 'EZ', 'HT', 'NF', 'SO', 'SD', 'PF']
        
        if use_transformers:
            self._load_transformer()
    
    def _load_transformer(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            # MiniLM is fast and produces good 384-dim embeddings
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Loaded sentence transformer for NLP features")
        except ImportError:
            print("⚠ sentence-transformers not installed, using TF-IDF fallback")
            self.use_transformers = False
    
    def extract_features(self, best_scores, recent_scores=None):
        """
        Extract all NLP features from player's scores.
        
        Args:
            best_scores: List of player's best score objects from osu! API
            recent_scores: Optional list of recent scores
            
        Returns:
            Dictionary of NLP features
        """
        features = {}
        
        # Combine scores for analysis
        all_scores = best_scores + (recent_scores or [])
        
        if not all_scores:
            return self._empty_features()
        
        # 1. Extract text data from beatmaps
        titles, artists, tags, diff_names, mods_list = self._extract_text_data(all_scores)
        
        # 2. Genre/style features
        genre_features = self._extract_genre_features(titles, artists, tags)
        features.update(genre_features)
        
        # 3. Difficulty preference features
        diff_features = self._extract_difficulty_features(diff_names)
        features.update(diff_features)
        
        # 4. Mod combination features (sequence-like)
        mod_features = self._extract_mod_features(mods_list)
        features.update(mod_features)
        
        # 5. Text diversity features
        diversity_features = self._extract_diversity_features(titles, artists)
        features.update(diversity_features)
        
        # 6. Optional: Deep embeddings from transformer
        if self.use_transformers and self.encoder:
            embedding_features = self._extract_embeddings(titles, tags)
            features.update(embedding_features)
        
        return features
    
    def _extract_text_data(self, scores):
        """Extract text fields from score objects"""
        titles = []
        artists = []
        tags = []
        diff_names = []
        mods_list = []
        
        for score in scores:
            beatmapset = score.get('beatmapset', {}) or {}
            beatmap = score.get('beatmap', {}) or {}
            
            # Title and artist
            title = beatmapset.get('title', '') or ''
            artist = beatmapset.get('artist', '') or ''
            tag_str = beatmapset.get('tags', '') or ''
            
            # Difficulty name (e.g., "Insane", "Expert")
            diff_name = beatmap.get('version', '') or ''
            
            # Mods used
            mods = score.get('mods', []) or []
            if isinstance(mods, list):
                # Handle both string mods and dict mods
                mod_strs = []
                for m in mods:
                    if isinstance(m, str):
                        mod_strs.append(m)
                    elif isinstance(m, dict):
                        mod_strs.append(m.get('acronym', ''))
                mods_list.append(mod_strs)
            
            titles.append(title.lower())
            artists.append(artist.lower())
            tags.append(tag_str.lower())
            diff_names.append(diff_name.lower())
        
        return titles, artists, tags, diff_names, mods_list
    
    def _extract_genre_features(self, titles, artists, tags):
        """Extract genre/style preference features"""
        features = {}
        
        # Combine all text for genre detection
        all_text = ' '.join(titles + artists + tags)
        
        # Count genre keyword matches
        total_matches = 0
        for genre, keywords in self.genre_keywords.items():
            count = sum(1 for kw in keywords if kw in all_text)
            features[f'nlp_genre_{genre}'] = count
            total_matches += count
        
        # Normalize by total matches
        if total_matches > 0:
            for genre in self.genre_keywords:
                features[f'nlp_genre_{genre}_ratio'] = features[f'nlp_genre_{genre}'] / total_matches
        else:
            for genre in self.genre_keywords:
                features[f'nlp_genre_{genre}_ratio'] = 0
        
        return features
    
    def _extract_difficulty_features(self, diff_names):
        """Extract difficulty preference features from difficulty names"""
        features = {}
        
        all_diff_text = ' '.join(diff_names)
        
        # Count difficulty level keywords
        total = 0
        for level, keywords in self.difficulty_keywords.items():
            count = sum(1 for kw in keywords if kw in all_diff_text)
            features[f'nlp_diff_{level}'] = count
            total += count
        
        # Calculate preference ratios
        if total > 0:
            for level in self.difficulty_keywords:
                features[f'nlp_diff_{level}_ratio'] = features[f'nlp_diff_{level}'] / total
        else:
            for level in self.difficulty_keywords:
                features[f'nlp_diff_{level}_ratio'] = 0
        
        # Custom difficulty name patterns (numbers often indicate star rating)
        number_pattern = re.compile(r'\d+')
        numbers_found = []
        for name in diff_names:
            matches = number_pattern.findall(name)
            numbers_found.extend([int(n) for n in matches if 1 <= int(n) <= 15])
        
        features['nlp_avg_diff_number'] = np.mean(numbers_found) if numbers_found else 0
        features['nlp_max_diff_number'] = max(numbers_found) if numbers_found else 0
        
        return features
    
    def _extract_mod_features(self, mods_list):
        """
        Extract features from mod combinations.
        Treats mod sequences like "sentences" in NLP.
        """
        features = {}
        
        # Flatten all mods
        all_mods = [mod for mods in mods_list for mod in mods]
        
        # Count each mod type
        mod_counts = Counter(all_mods)
        for mod in self.mod_vocab:
            features[f'nlp_mod_{mod.lower()}'] = mod_counts.get(mod, 0)
        
        # Mod combination patterns (bigrams)
        bigrams = []
        for mods in mods_list:
            if len(mods) >= 2:
                sorted_mods = sorted(mods)  # Sort for consistency
                for i in range(len(sorted_mods) - 1):
                    bigrams.append(f"{sorted_mods[i]}_{sorted_mods[i+1]}")
        
        # Common mod combinations
        bigram_counts = Counter(bigrams)
        features['nlp_mod_hdhr'] = bigram_counts.get('HD_HR', 0)
        features['nlp_mod_hddt'] = bigram_counts.get('DT_HD', 0) + bigram_counts.get('HD_NC', 0)
        features['nlp_mod_hrdt'] = bigram_counts.get('DT_HR', 0) + bigram_counts.get('HR_NC', 0)
        
        # Mod diversity (vocabulary size)
        unique_mods = len(set(all_mods))
        features['nlp_mod_diversity'] = unique_mods
        features['nlp_mod_total'] = len(all_mods)
        features['nlp_avg_mods_per_play'] = len(all_mods) / max(len(mods_list), 1)
        
        # No-mod ratio
        nomod_count = sum(1 for mods in mods_list if len(mods) == 0)
        features['nlp_nomod_ratio'] = nomod_count / max(len(mods_list), 1)
        
        return features
    
    def _extract_diversity_features(self, titles, artists):
        """Extract text diversity features"""
        features = {}
        
        # Unique titles/artists ratio (vocabulary diversity)
        unique_titles = len(set(titles))
        unique_artists = len(set(artists))
        
        features['nlp_unique_songs'] = unique_titles
        features['nlp_unique_artists'] = unique_artists
        features['nlp_song_diversity'] = unique_titles / max(len(titles), 1)
        features['nlp_artist_diversity'] = unique_artists / max(len(artists), 1)
        
        # Average title length (complexity indicator)
        title_lengths = [len(t.split()) for t in titles]
        features['nlp_avg_title_length'] = np.mean(title_lengths) if title_lengths else 0
        
        # Language detection heuristic (Japanese characters)
        japanese_count = sum(1 for t in titles if self._has_japanese(t))
        features['nlp_japanese_ratio'] = japanese_count / max(len(titles), 1)
        
        return features
    
    def _has_japanese(self, text):
        """Check if text contains Japanese characters"""
        # Hiragana, Katakana, or common Kanji ranges
        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
        return bool(japanese_pattern.search(text))
    
    def _extract_embeddings(self, titles, tags):
        """Extract deep embeddings using sentence transformer"""
        features = {}
        
        # Combine text for embedding
        combined_titles = ' '.join(titles[:50])  # Limit for efficiency
        combined_tags = ' '.join(tags[:50])
        
        try:
            # Get embeddings (384 dimensions each for MiniLM)
            title_emb = self.encoder.encode(combined_titles)
            
            # Use first N dimensions to avoid feature explosion
            n_dims = 32  # Reduced dimensionality
            for i in range(min(n_dims, len(title_emb))):
                features[f'nlp_emb_{i}'] = float(title_emb[i])
            
            # Also get embedding statistics
            features['nlp_emb_mean'] = float(np.mean(title_emb))
            features['nlp_emb_std'] = float(np.std(title_emb))
            features['nlp_emb_norm'] = float(np.linalg.norm(title_emb))
            
        except Exception as e:
            print(f"Warning: Embedding extraction failed: {e}")
            # Fill with zeros
            for i in range(32):
                features[f'nlp_emb_{i}'] = 0
            features['nlp_emb_mean'] = 0
            features['nlp_emb_std'] = 0
            features['nlp_emb_norm'] = 0
        
        return features
    
    def _empty_features(self):
        """Return empty features when no scores available"""
        features = {}
        
        # Genre features
        for genre in self.genre_keywords:
            features[f'nlp_genre_{genre}'] = 0
            features[f'nlp_genre_{genre}_ratio'] = 0
        
        # Difficulty features
        for level in self.difficulty_keywords:
            features[f'nlp_diff_{level}'] = 0
            features[f'nlp_diff_{level}_ratio'] = 0
        features['nlp_avg_diff_number'] = 0
        features['nlp_max_diff_number'] = 0
        
        # Mod features
        for mod in self.mod_vocab:
            features[f'nlp_mod_{mod.lower()}'] = 0
        features['nlp_mod_hdhr'] = 0
        features['nlp_mod_hddt'] = 0
        features['nlp_mod_hrdt'] = 0
        features['nlp_mod_diversity'] = 0
        features['nlp_mod_total'] = 0
        features['nlp_avg_mods_per_play'] = 0
        features['nlp_nomod_ratio'] = 1
        
        # Diversity features
        features['nlp_unique_songs'] = 0
        features['nlp_unique_artists'] = 0
        features['nlp_song_diversity'] = 0
        features['nlp_artist_diversity'] = 0
        features['nlp_avg_title_length'] = 0
        features['nlp_japanese_ratio'] = 0
        
        # Embedding features (if using transformers)
        if self.use_transformers:
            for i in range(32):
                features[f'nlp_emb_{i}'] = 0
            features['nlp_emb_mean'] = 0
            features['nlp_emb_std'] = 0
            features['nlp_emb_norm'] = 0
        
        return features


# Convenience function for quick feature extraction
def extract_nlp_features(best_scores, recent_scores=None, use_transformers=False):
    """
    Quick function to extract NLP features from scores.
    
    Args:
        best_scores: Player's best scores from osu! API
        recent_scores: Player's recent scores (optional)
        use_transformers: Use deep learning embeddings
        
    Returns:
        Dictionary of NLP features
    """
    extractor = NLPFeatureExtractor(use_transformers=use_transformers)
    return extractor.extract_features(best_scores, recent_scores)
