"""
Ticker Selection Pipeline - main orchestrator.

Executes the five-stage pipeline:
  Stage 0: Universe discovery (Alpaca get_all_assets)
  Stage 1: Snapshot screening (price, volume, ATR)
  Stage 2: STRAT pattern scan (reuses PaperSignalScanner)
  Stage 3: TFC filter (already in scanner pipeline)
  Stage 4: Composite scoring + ranking

Output: data/candidates/candidates.json
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from strat.ticker_selection.config import TickerSelectionConfig
from strat.ticker_selection.convergence import ConvergenceAnalyzer, ConvergenceMetadata
from strat.ticker_selection.universe import UniverseManager
from strat.ticker_selection.screener import SnapshotScreener, ScreenedStock
from strat.ticker_selection.scorer import CandidateScorer, ScoredCandidate

logger = logging.getLogger(__name__)


class TickerSelectionPipeline:
    """
    Orchestrates the full ticker selection pipeline.

    Reuses PaperSignalScanner for STRAT pattern detection and TFC
    evaluation -- zero reimplementation of the 5,939 lines of
    validated detection code.
    """

    def __init__(self, config: Optional[TickerSelectionConfig] = None):
        self.config = config or TickerSelectionConfig()
        self.universe_mgr = UniverseManager(self.config)
        self.screener = SnapshotScreener(self.config)
        self.scorer = CandidateScorer(self.config)
        self._scanner = None

    def _get_scanner(self):
        """Lazy-init PaperSignalScanner (heavy imports)."""
        if self._scanner is None:
            from strat.paper_signal_scanner import PaperSignalScanner
            self._scanner = PaperSignalScanner()
        return self._scanner

    def run(self, dry_run: bool = False) -> Dict:
        """
        Execute the full pipeline.

        Args:
            dry_run: If True, run all stages but don't write candidates.json.

        Returns:
            Pipeline results dict (same schema as candidates.json).
        """
        start = time.time()
        stats = {
            'universe_size': 0,
            'screened_size': 0,
            'patterns_found': 0,
            'tfc_qualified': 0,
            'unique_symbols': 0,
            'final_candidates': 0,
            'scan_duration_seconds': 0.0,
        }

        # Stage 0: Universe discovery
        logger.info("Stage 0: Discovering universe...")
        universe = self.universe_mgr.get_universe()
        stats['universe_size'] = len(universe)
        logger.info(f"  Universe: {len(universe)} symbols")

        # Stage 1: Snapshot screening
        logger.info("Stage 1: Snapshot screening...")
        screened = self.screener.screen(universe)
        stats['screened_size'] = len(screened)
        logger.info(f"  Screened: {len(screened)} symbols passed")

        # Build lookup for screened data
        screened_map: Dict[str, ScreenedStock] = {s.symbol: s for s in screened}

        # Stage 2+3: STRAT scan + TFC (via PaperSignalScanner)
        logger.info("Stage 2: STRAT pattern scanning + TFC evaluation...")
        scanner = self._get_scanner()
        candidates_raw: List[Dict] = []
        error_count = 0

        # Pre-fetch VIX once so every scan reuses the cached value
        try:
            vix_value = scanner.prefetch_vix()
            logger.info(f"  VIX pre-fetched: {vix_value:.1f}")
        except Exception as e:
            logger.warning(f"  VIX pre-fetch failed: {e}")

        # Cap symbols sent to the expensive STRAT scanner
        # Screened list is already sorted by dollar volume desc
        scan_symbols = [s.symbol for s in screened[:self.config.max_scan_symbols]]
        logger.info(
            f"  Scanning top {len(scan_symbols)} of {len(screened)} "
            f"screened symbols (by dollar volume)"
        )
        def _scan_single(sym):
            """Scan a single symbol, returning (symbol, signals, error)."""
            try:
                sigs = scanner.scan_symbol_all_timeframes_resampled(sym)
                return sym, sigs, None
            except Exception as exc:
                return sym, [], exc

        max_workers = min(self.config.max_workers, len(scan_symbols)) if scan_symbols else 1
        completed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_scan_single, sym): sym
                for sym in scan_symbols
            }
            for future in as_completed(futures):
                symbol, signals, scan_error = future.result()
                completed_count += 1
                if completed_count % 25 == 0:
                    logger.info(f"  Scanning {completed_count}/{len(scan_symbols)}...")

                if scan_error:
                    error_count += 1
                    logger.debug(f"  Scan error for {symbol}: {scan_error}")
                    continue

                for sig in signals:
                    # Hard filter: Only SETUP signals (waiting for break)
                    if sig.signal_type != 'SETUP':
                        continue

                    # Hard filter: TFC must pass flexible check
                    if not sig.context.tfc_passes:
                        continue

                    # Hard filter: TFC direction must match pattern direction
                    tfc_dir = 'bullish' if sig.direction == 'CALL' else 'bearish'
                    if sig.context.tfc_score > 0:
                        # TFC direction comes from the alignment label
                        tfc_label = sig.context.tfc_alignment.upper()
                        if tfc_dir.upper() not in tfc_label:
                            continue

                    stock_data = screened_map.get(symbol)
                    candidates_raw.append({
                        'signal': sig,
                        'stock': stock_data,
                    })

        stats['patterns_found'] = len(candidates_raw)
        # TFC qualified is the same as patterns_found since we hard-filter above
        stats['tfc_qualified'] = len(candidates_raw)
        logger.info(
            f"  Patterns found (TFC qualified): {len(candidates_raw)} "
            f"({error_count} errors)"
        )

        # Stage 4: Scoring + ranking
        logger.info("Stage 4: Scoring and ranking...")
        scored = self._score_candidates(candidates_raw, screened_map)

        # Dedup: keep highest-scoring candidate per symbol
        deduped = self._dedup_by_symbol(scored)
        stats['unique_symbols'] = len(deduped)
        logger.info(
            f"  Dedup: {len(scored)} scored -> {len(deduped)} unique symbols"
        )

        # Convergence analysis on deduped symbols (~40, not 100)
        convergence_map = self._analyze_convergence(deduped)

        # Tier classification: T1 (convergence), T2 (standard), T3 (continuation)
        self._classify_tiers(deduped, convergence_map)

        # Tiered sort + per-tier caps
        final = self._sort_and_cap_tiered(deduped, convergence_map)

        # Assign ranks across all tiers
        for i, c in enumerate(final):
            c.rank = i + 1

        tier_counts = {1: 0, 2: 0, 3: 0}
        for c in final:
            tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1
        stats['tier1_count'] = tier_counts[1]
        stats['tier2_count'] = tier_counts[2]
        stats['tier3_count'] = tier_counts[3]
        stats['final_candidates'] = len(final)
        duration = time.time() - start
        stats['scan_duration_seconds'] = round(duration, 1)

        logger.info(
            f"Pipeline complete: {len(final)} candidates in {duration:.1f}s"
        )

        # Finviz enrichment (informational only)
        enrichment_map = {}
        if self.config.finviz_enrichment_enabled:
            enrichment_map = self._enrich_candidates(final)

        # Build output
        result = self._build_output(final, stats, enrichment_map)

        # Write to disk
        if not dry_run:
            self._write_candidates(result)
            self._send_discord_summary(final, stats, enrichment_map)

        return result

    @staticmethod
    def _earliest_tradeable_time(timeframe: str, pattern_type: str) -> str:
        """Compute when a setup becomes tradeable based on timeframe and bar count.

        Daily/Weekly/Monthly setups are tradeable at market open (09:30 ET)
        because all bars are already closed. Hourly setups must wait for
        today's bars to close:
          2-bar patterns: 10:30 ET (first 1H bar closes)
          3-bar patterns: 11:30 ET (second 1H bar closes)

        Returns time string in "HH:MM ET" format.
        """
        if timeframe != '1H':
            return '09:30 ET'

        is_3bar = len(pattern_type.split('-')) >= 3
        return '11:30 ET' if is_3bar else '10:30 ET'

    @staticmethod
    def _dedup_by_symbol(
        scored: List[ScoredCandidate],
    ) -> List[ScoredCandidate]:
        """Keep only the highest-scoring candidate per symbol.

        When a symbol has multiple setups (different timeframes or patterns),
        only the best composite_score is retained. This prevents the morning
        report from showing e.g. GOOGL x3 across 1H/1D/1W.
        """
        best: Dict[str, ScoredCandidate] = {}
        for c in scored:
            existing = best.get(c.symbol)
            if existing is None or c.composite_score > existing.composite_score:
                best[c.symbol] = c
        return list(best.values())

    def _analyze_convergence(
        self, candidates: List[ScoredCandidate],
    ) -> Dict[str, ConvergenceMetadata]:
        """Run multi-TF convergence analysis on deduped candidate symbols."""
        symbols = [c.symbol for c in candidates]
        prices = {c.symbol: c.current_price for c in candidates}

        if not symbols:
            return {}

        try:
            analyzer = ConvergenceAnalyzer(
                alpaca_account=self.config.alpaca_account,
            )
            result = analyzer.analyze_batch(symbols, prices)
            convergence_count = sum(
                1 for m in result.values() if m.is_convergence
            )
            logger.info(
                f"  Convergence: {len(result)} analyzed, "
                f"{convergence_count} with multi-TF inside bars"
            )
            return result
        except Exception as e:
            logger.warning(f"Convergence analysis failed: {e}")
            return {}

    def _classify_tiers(
        self,
        candidates: List[ScoredCandidate],
        convergence_map: Dict[str, ConvergenceMetadata],
    ) -> None:
        """
        Assign tier 1/2/3 to each candidate in-place.

        Tier 1: Multi-TF convergence (2+ inside bars across M/W/D)
        Tier 2: Standard setups (default)
        Tier 3: Continuation context (same-direction bar sequence + entry)
        """
        min_inside = self.config.convergence_min_inside_tfs

        for c in candidates:
            meta = convergence_map.get(c.symbol)
            if meta is not None:
                c.convergence = meta

            # Tier 1: convergence detected
            if meta is not None and meta.inside_bar_count >= min_inside:
                c.tier = 1
                continue

            # Tier 3: true continuation
            if self.scorer._is_continuation(c.pattern_type, c.direction):
                c.tier = 3
                continue

            # Tier 2: everything else (default)
            c.tier = 2

    def _sort_and_cap_tiered(
        self,
        candidates: List[ScoredCandidate],
        convergence_map: Dict[str, ConvergenceMetadata],
    ) -> List[ScoredCandidate]:
        """
        Sort candidates within each tier and apply per-tier caps.

        Tier 1: sorted by convergence_score desc, capped at max_tier1
        Tier 2: sorted by composite_score desc, capped at max_tier2
        Tier 3: sorted by composite_score desc, capped at max_tier3

        Returns combined list: T1 first, then T2, then T3.
        """
        tier1 = [c for c in candidates if c.tier == 1]
        tier2 = [c for c in candidates if c.tier == 2]
        tier3 = [c for c in candidates if c.tier == 3]

        # Tier 1: sort by convergence_score (structural quality)
        tier1.sort(
            key=lambda c: (
                convergence_map.get(c.symbol, ConvergenceMetadata(
                    inside_bar_count=0, inside_bar_timeframes=[],
                    trigger_levels={}, convergence_score=0.0,
                    bullish_trigger=None, bearish_trigger=None,
                    trigger_spread_pct=0.0, prior_direction_alignment='mixed',
                    is_convergence=False,
                )).convergence_score
            ),
            reverse=True,
        )

        # Tier 2 and 3: sort by composite_score
        tier2.sort(key=lambda c: c.composite_score, reverse=True)
        tier3.sort(key=lambda c: c.composite_score, reverse=True)

        # Apply caps
        cfg = self.config
        tier1 = tier1[:cfg.max_tier1_candidates]
        tier2 = tier2[:cfg.max_tier2_candidates]
        tier3 = tier3[:cfg.max_tier3_context]

        logger.info(
            f"  Tiers: T1={len(tier1)} T2={len(tier2)} T3={len(tier3)} "
            f"(caps: {cfg.max_tier1_candidates}/{cfg.max_tier2_candidates}/{cfg.max_tier3_context})"
        )

        return tier1 + tier2 + tier3

    def _score_candidates(
        self,
        candidates_raw: List[Dict],
        screened_map: Dict[str, ScreenedStock],
    ) -> List[ScoredCandidate]:
        """Score all candidates using CandidateScorer."""
        scored = []
        for entry in candidates_raw:
            sig = entry['signal']
            stock: Optional[ScreenedStock] = entry.get('stock')

            current_price = stock.price if stock else sig.entry_trigger
            atr_pct = stock.atr_percent if stock else sig.context.atr_percent
            dv = stock.dollar_volume if stock else 0.0

            candidate = self.scorer.score(
                symbol=sig.symbol,
                pattern_type=sig.pattern_type,
                signal_type=sig.signal_type,
                direction=sig.direction,
                timeframe=sig.timeframe,
                is_bidirectional=getattr(sig, 'is_bidirectional', True),
                entry_trigger=sig.entry_trigger,
                stop_price=sig.stop_price,
                target_price=sig.target_price,
                current_price=current_price,
                tfc_score=sig.context.tfc_score,
                tfc_alignment=sig.context.tfc_alignment,
                tfc_direction='bullish' if sig.direction == 'CALL' else 'bearish',
                tfc_passes_flexible=sig.context.tfc_passes,
                tfc_risk_multiplier=sig.context.risk_multiplier,
                tfc_priority_rank=sig.context.priority_rank,
                atr_percent=atr_pct,
                dollar_volume=dv,
            )
            scored.append(candidate)
        return scored

    def _enrich_candidates(
        self, candidates: List[ScoredCandidate]
    ) -> Dict:
        """Enrich candidates with Finviz data. Returns {} on any error."""
        try:
            from strat.ticker_selection.enrichment import FinvizEnricher
        except ImportError:
            logger.warning("Enrichment module unavailable, skipping")
            return {}

        try:
            symbols = list(dict.fromkeys(c.symbol for c in candidates))
            enricher = FinvizEnricher(
                cache_ttl=self.config.finviz_cache_ttl_hours * 3600,
                max_workers=self.config.finviz_max_workers,
            )
            result = enricher.enrich_candidates(symbols)
            logger.info(f"Finviz enrichment: {len(result)} symbols enriched")
            return result
        except Exception as e:
            logger.warning(f"Finviz enrichment failed: {e}")
            return {}

    def _build_output(self, candidates: List[ScoredCandidate], stats: Dict, enrichment_map: Optional[Dict] = None) -> Dict:
        """Build the candidates.json output dict."""
        now = datetime.now(timezone.utc).isoformat()

        output = {
            'version': '2.0',
            'generated_at': now,
            'pipeline_stats': stats,
            'core_symbols': list(self.config.core_symbols),
            'candidates': [],
        }

        for c in candidates:
            entry = {
                'symbol': c.symbol,
                'composite_score': c.composite_score,
                'rank': c.rank,
                'tier': c.tier,
                'pattern': {
                    'type': c.pattern_type,
                    'base_type': c.base_pattern,
                    'signal_type': c.signal_type,
                    'direction': c.direction,
                    'timeframe': c.timeframe,
                    'is_bidirectional': c.is_bidirectional,
                },
                'levels': {
                    'entry_trigger': c.entry_trigger,
                    'stop_price': c.stop_price,
                    'target_price': c.target_price,
                    'current_price': c.current_price,
                    'distance_to_trigger_pct': c.distance_to_trigger_pct,
                },
                'tfc': {
                    'score': c.tfc_score,
                    'alignment': c.tfc_alignment,
                    'direction': c.tfc_direction,
                    'passes_flexible': c.tfc_passes_flexible,
                    'risk_multiplier': c.tfc_risk_multiplier,
                    'priority_rank': c.tfc_priority_rank,
                },
                'metrics': {
                    'atr_percent': c.atr_percent,
                    'dollar_volume': c.dollar_volume,
                    'risk_reward': c.risk_reward,
                },
                'scoring_breakdown': {
                    'tfc_component': c.breakdown.tfc_component,
                    'pattern_component': c.breakdown.pattern_component,
                    'proximity_component': c.breakdown.proximity_component,
                    'atr_component': c.breakdown.atr_component,
                },
            }
            if c.convergence is not None:
                entry['convergence'] = c.convergence.to_dict()
            if enrichment_map and c.symbol in enrichment_map:
                entry['finviz'] = enrichment_map[c.symbol].to_dict()
            output['candidates'].append(entry)

        return output

    def _write_candidates(self, data: Dict) -> None:
        """Write candidates.json atomically (write .tmp then os.replace)."""
        path = Path(self.config.candidates_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.with_suffix('.json.tmp')
        try:
            tmp_path.write_text(json.dumps(data, indent=2))
            os.replace(str(tmp_path), str(path))
            logger.info(f"Candidates written: {path}")
        except Exception as e:
            logger.error(f"Failed to write candidates: {e}")
            # Clean up temp file
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _send_discord_summary(
        self,
        candidates: List[ScoredCandidate],
        stats: Dict,
        enrichment_map: Optional[Dict] = None,
    ) -> None:
        """Send a summary of top candidates to Discord."""
        try:
            webhook_url = (
                os.environ.get('DISCORD_EQUITY_WEBHOOK_URL')
                or os.environ.get('DISCORD_WEBHOOK_URL')
            )
            if not webhook_url:
                logger.debug("No Discord webhook configured for ticker selection")
                return

            # Build summary message
            lines = [
                f"**Ticker Selection Pipeline Complete**",
                f"Universe: {stats['universe_size']:,} | "
                f"Screened: {stats['screened_size']} | "
                f"Patterns: {stats['patterns_found']} | "
                f"Final: {stats['final_candidates']}",
                f"Duration: {stats['scan_duration_seconds']}s",
                "",
            ]

            if candidates:
                lines.append("**Top Candidates:**")
                for c in candidates[:8]:
                    direction_label = '[CALL]' if c.direction == 'CALL' else '[PUT]'
                    lines.append(
                        f"{direction_label} **{c.symbol}** {c.pattern_type} {c.timeframe} "
                        f"| Score: {c.composite_score} "
                        f"| TFC: {c.tfc_alignment} "
                        f"| ATR: {c.atr_percent}%"
                    )
                    # Append compact Finviz context if available
                    if enrichment_map and c.symbol in enrichment_map:
                        e = enrichment_map[c.symbol]
                        parts = []
                        if e.sector:
                            parts.append(e.sector)
                        if e.earnings_date:
                            parts.append(f"Earn: {e.earnings_date}")
                        if e.target_price is not None:
                            parts.append(f"Tgt: ${e.target_price:.0f}")
                        if e.analyst_recommendation:
                            parts.append(f"Rec: {e.analyst_recommendation}")
                        if parts:
                            lines.append(f"  _({' | '.join(parts)})_")
            else:
                lines.append("_No candidates qualified today._")

            message = '\n'.join(lines)

            import requests
            resp = requests.post(
                webhook_url,
                json={'content': message},
                timeout=10,
            )
            if resp.status_code in (200, 204):
                logger.info("Discord ticker selection summary sent")
            else:
                logger.warning(f"Discord send failed: {resp.status_code}")

        except Exception as e:
            logger.warning(f"Failed to send Discord summary: {e}")


def run_selection(dry_run: bool = False) -> Dict:
    """
    Convenience function for running the pipeline (e.g., from cron).

    Loads config from environment and runs the full pipeline.
    """
    config = TickerSelectionConfig.from_env()
    pipeline = TickerSelectionPipeline(config)
    return pipeline.run(dry_run=dry_run)
