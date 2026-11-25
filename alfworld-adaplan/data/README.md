# 실험 데이터 (2026-11-25 기준)

- `episodes.csv`: 에피소드별 결과 요약 5691행 (sid, sig, task, rep, ttype, success, steps, expansions, reward, in_tokens, out_tokens, cost_usd). 계획·행동 기록 제외. `sig`와 `ttype`으로 집합을 구분한다(초기 행은 sig·rep가 비어 있음).
- `results_cache.jsonl.gz`: 원본 캐시(계획·행동 기록 포함, 압축). `gunzip -k`로 풀어 `results_cache.jsonl`로 쓴다.
- `results_cache_archive.jsonl.gz`: 보관함으로 옮긴 이전 캐시 행(압축).
- `cost_calls.jsonl.gz`: 호출별 비용 로그(압축).
- 분석: `../adapt_analysis.py`(최종 통합 스크립트)와 `../analysis_scripts/`(작업 중 쓴 개별 스크립트). 숫자는 `../EXPERIMENT_LOG.md` (45)~(55) 참고.
