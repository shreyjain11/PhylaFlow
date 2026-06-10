import sys, re, math
# Usage: python check_ess.py <pstat_file> <ess_threshold>
pstat, thresh = sys.argv[1], float(sys.argv[2])

min_ess = math.inf
avg_ess = math.inf

with open(pstat, 'r', encoding='utf-8', errors='ignore') as f:
    in_table = False
    for line in f:
        if re.match(r'\s*Parameter\s+Mean\s+Variance', line):
            in_table = True
            continue
        if in_table and 'TL' in line:
            if line.strip().startswith('---') or not line.strip():
                continue
            # lines like: "TL  5.5 ...   min ESS  avg ESS  PSRF"
            cols = line.strip().split()
            if len(cols) < 8:  # protect against footer lines
                continue
            try:
                # Last 3 numeric columns are min ESS, avg ESS, PSRF
                minE = float(cols[-3])
                avgE = float(cols[-2])
                min_ess = min(min_ess, minE)
                avg_ess = min(avg_ess, avgE)
            except:
                pass

ok = (min_ess >= thresh) and (avg_ess >= thresh)
print(f"MIN_ESS for TL={min_ess:.2f} AVG_ESS for TL={avg_ess:.2f} THRESH={thresh} OK={ok}")
sys.exit(0 if ok else 1)