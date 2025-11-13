# Lagrs Development Roadmap

## Current Status ✅

- ✅ Parallel rolling mean (Rayon)
- ✅ ARIMA(p,d,q) with MLE estimation
- ✅ Parallel batch processing for multi-SKU
- ✅ Zero-copy Python-Rust data transfer
- ✅ Comprehensive test suite
- ✅ GitHub Actions CI/CD

## Roadmap Analysis & Recommendations

### 🎯 **Phase 1: Fast Rolling Ops** (RECOMMENDED - Quick Win)

**Status**: Partially done (rolling mean exists)

**What to add:**
- ✅ Rolling mean (done)
- ⏳ Rolling std
- ⏳ Rolling sum
- ⏳ Windowed min/max
- ⏳ Rolling median (more complex, but useful)

**Timeline**: 2-3 days (not 3-7)

**Why this is good:**
- ✅ Quick wins build momentum
- ✅ Easy to benchmark and show value
- ✅ Foundation for other features
- ✅ Low risk, high visibility

**Recommendation**: **DO THIS FIRST** - It's almost done, just needs a few more functions.

---

### ⚙️ **Phase 2: State Space Foundation** (CONSIDER DELAYING)

**Timeline**: 2-4 weeks (realistic)

**Issues:**
- ❌ **Too early** - You don't need this for ARIMA
- ❌ **Over-engineering** - ARIMA can work without full state-space
- ❌ **Complexity** - Kalman filter is non-trivial
- ✅ **Future-proof** - But not urgent

**Recommendation**: **SKIP FOR NOW** - Come back after Phase 3 if you need SARIMAX/VARMAX. Current ARIMA implementation works fine without it.

**Better alternative**: Focus on making ARIMA robust first, then add state-space later if needed.

---

### 🎯 **Phase 3: ARIMA v0.1 Complete** (IN PROGRESS - PRIORITIZE)

**Status**: 70% done

**What's done:**
- ✅ ARIMA(p,d,q) basic implementation
- ✅ MLE estimation (gradient descent)
- ✅ Differencing
- ✅ Forecasting
- ✅ AIC/BIC

**What's missing:**
- ⏳ SARIMA(P,D,Q,s) - seasonal ARIMA
- ⏳ Better optimization (BFGS instead of gradient descent)
- ⏳ Parameter constraints (stationarity/invertibility)
- ⏳ Confidence intervals for forecasts
- ⏳ Model diagnostics (residual analysis)

**Timeline**: 1-2 weeks (not 2-3)

**Recommendation**: **COMPLETE THIS** - It's your core differentiator. Add:
1. SARIMA support (high value)
2. Better optimization (accuracy)
3. Confidence intervals (production-ready)

---

### 🧠 **Phase 4: Auto Model Selection** (HIGH VALUE)

**Timeline**: 1-2 weeks (realistic)

**Why this matters:**
- ✅ **Huge user value** - Everyone wants auto-arima
- ✅ **Differentiator** - pmdarima is slow
- ✅ **Parallelizable** - Perfect for Rust

**Recommendation**: **DO THIS** - But simplify:
- Start with simple grid search (p,d,q only)
- Add seasonal later
- Use parallel evaluation (Rayon)
- Cache results

**Success metric**: 10-50x faster than pmdarima (achievable with parallelization)

---

### 🌙 **Phase 5: ETS / Holt-Winters** (MEDIUM PRIORITY)

**Timeline**: 1-2 weeks (realistic)

**Why it's good:**
- ✅ Popular for retail/forecasting
- ✅ Simpler than ARIMA
- ✅ Good complement

**Recommendation**: **DO THIS** - But after Phase 4. ETS is simpler and faster to implement.

---

### 🔮 **Phase 6: Seasonality Detection** (MEDIUM PRIORITY)

**Timeline**: 1-2 weeks (realistic)

**Why it matters:**
- ✅ Enables better auto-selection
- ✅ Useful for Prophet-like features
- ✅ STL is well-understood

**Recommendation**: **CONSIDER** - But can be done incrementally. Start with simple periodogram detection, add STL later.

---

### 💥 **Phase 7: Gradient-Boosted TS** (LOW PRIORITY - TOO AMBITIOUS)

**Timeline**: 3-6 weeks (probably 8-12 weeks realistically)

**Issues:**
- ❌ **Huge scope** - This is basically building XGBoost
- ❌ **Competition** - XGBoost, LightGBM already exist
- ❌ **Different problem** - Not time-series specific
- ❌ **Complexity** - Tree algorithms are non-trivial

**Recommendation**: **SKIP OR DELAY** - Focus on time-series specific models first. Users can use XGBoost with lag features if needed.

**Alternative**: Build a **feature engineering** library that generates lag/rolling features for XGBoost instead.

---

### 🔁 **Phase 8: Cross-Validation** (HIGH VALUE)

**Timeline**: 1 week (realistic)

**Why it matters:**
- ✅ **Essential for production** - Everyone needs backtesting
- ✅ **Easy to parallelize** - Perfect for Rust
- ✅ **Differentiator** - Fast backtesting is valuable

**Recommendation**: **DO THIS EARLY** - It's relatively simple and high value. Can be done after Phase 3.

---

### 🏗️ **Phase 9: Hierarchical Forecasting** (MEDIUM PRIORITY)

**Timeline**: 2-4 weeks (realistic)

**Why it matters:**
- ✅ **Enterprise need** - Many companies need this
- ✅ **Niche** - Less competition
- ✅ **Scalable** - Rust can handle large hierarchies

**Recommendation**: **CONSIDER** - But after core models are solid. This is advanced.

---

### ☀️ **Phase 10: Production Features** (ONGOING)

**Timeline**: Continuous

**What's needed:**
- ✅ Model serialization (save/load)
- ✅ Arrow/Parquet I/O
- ✅ Better error handling
- ✅ Logging
- ✅ Documentation

**Recommendation**: **DO INCREMENTALLY** - Add these as you go, not as a separate phase.

---

### 🧊 **Phase 11: GPU Acceleration** (OPTIONAL - LATER)

**Timeline**: 4-8 weeks (probably longer)

**Issues:**
- ❌ **Huge investment** - CUDA is complex
- ❌ **Limited benefit** - Most users don't have GPUs
- ❌ **Maintenance burden** - GPU code is hard to maintain

**Recommendation**: **SKIP FOR NOW** - Focus on CPU parallelism first. GPU can come much later if there's demand.

---

### 🧩 **Phase 12: Documentation** (ONGOING)

**Recommendation**: **DO CONTINUOUSLY** - Don't wait. Document as you build.

---

## Revised Recommended Roadmap

### **Immediate (Next 2-4 weeks)**
1. ✅ **Complete Phase 1** - Add rolling std, sum, min/max (2-3 days)
2. ✅ **Complete Phase 3** - SARIMA, better optimization, confidence intervals (1-2 weeks)
3. ✅ **Phase 8** - Cross-validation/backtesting (1 week)

### **Short-term (1-2 months)**
4. ✅ **Phase 4** - Auto model selection (1-2 weeks)
5. ✅ **Phase 5** - ETS models (1-2 weeks)
6. ✅ **Phase 10** - Production features (ongoing)

### **Medium-term (3-6 months)**
7. ⏳ **Phase 6** - Seasonality detection (1-2 weeks)
8. ⏳ **Phase 9** - Hierarchical forecasting (2-4 weeks, if needed)

### **Later (If needed)**
9. ⏳ **Phase 2** - State-space foundation (only if SARIMAX/VARMAX needed)
10. ⏳ **Phase 7** - Gradient boosting (probably skip, use XGBoost instead)
11. ⏳ **Phase 11** - GPU (much later, if at all)

## Key Principles

1. **Focus on time-series specific features** - Don't rebuild general ML
2. **Leverage parallelism** - That's your advantage
3. **Incremental value** - Each phase should deliver usable features
4. **Realistic timelines** - ChatGPT's estimates are optimistic
5. **User feedback** - Build what users actually need

## What Makes lagrs Unique

1. **Speed** - Rust + parallelism = 10-100x faster
2. **Multi-SKU** - Parallel batch processing (unique!)
3. **Zero-copy** - Efficient Python integration
4. **Production-ready** - Fast enough for real-time use

Focus on these strengths rather than trying to build everything.

