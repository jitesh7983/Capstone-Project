import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import GaugeChart from "react-gauge-chart";
import { Bar, Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
} from "chart.js";

ChartJS.register(
  BarElement,
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

// ─── API URL ─────────────────────────────────────────────────────────────────
const API = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

// ─── AQI category config ─────────────────────────────────────────────────────
const CATEGORY_CONFIG = {
  Good:         { color: "#00c853", bg: "#e8f5e9", icon: "😊" },
  Satisfactory: { color: "#76c442", bg: "#f1f8e9", icon: "🙂" },
  Moderate:     { color: "#ffc107", bg: "#fffde7", icon: "😐" },
  Poor:         { color: "#ff6f00", bg: "#fff3e0", icon: "😷" },
  "Very Poor":  { color: "#e53935", bg: "#ffebee", icon: "🤢" },
  Severe:       { color: "#7b1fa2", bg: "#f3e5f5", icon: "☠️" }
};

// ─── Health advice (used in forecast peak alert) ──────────────────────────────
const HEALTH_ADVICE = {
  Good:         "Air quality is satisfactory. Enjoy outdoor activities freely.",
  Satisfactory: "Acceptable air quality. Sensitive individuals should limit prolonged exertion.",
  Moderate:     "Sensitive groups should reduce outdoor activity.",
  Poor:         "Everyone may experience health effects. Wear a mask outdoors.",
  "Very Poor":  "Health alert — avoid prolonged outdoor activity. Keep windows closed.",
  Severe:       "Health emergency. Stay indoors with air purification."
};

// ─── Pollutant meta ──────────────────────────────────────────────────────────
const POLLUTANT_META = {
  "PM2.5": { unit: "µg/m³", hint: "0 – 500", placeholder: "e.g. 45"  },
  "PM10":  { unit: "µg/m³", hint: "0 – 600", placeholder: "e.g. 80"  },
  "NO2":   { unit: "µg/m³", hint: "0 – 400", placeholder: "e.g. 30"  },
  "SO2":   { unit: "µg/m³", hint: "0 – 500", placeholder: "e.g. 15"  },
  "CO":    { unit: "mg/m³", hint: "0 – 50",  placeholder: "e.g. 1.2" },
  "O3":    { unit: "µg/m³", hint: "0 – 400", placeholder: "e.g. 55"  }
};

const EMPTY_FORM = {
  "PM2.5": "", "PM10": "", "NO2": "", "SO2": "", "CO": "", "O3": ""
};

// ─── Styles ───────────────────────────────────────────────────────────────────
const styles = {
  root: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%)",
    padding: "24px 16px",
    fontFamily: "'Sora', 'Segoe UI', sans-serif"
  },
  card: {
    background: "rgba(255,255,255,0.97)",
    borderRadius: "24px",
    boxShadow: "0 24px 64px rgba(0,0,0,0.35)",
    maxWidth: "1160px",
    margin: "0 auto",
    overflow: "hidden"
  },
  header: {
    background: "linear-gradient(135deg, #0f2027, #2c5364)",
    padding: "36px 40px 28px",
    color: "white",
    position: "relative",
    overflow: "hidden"
  },
  headerGlow: {
    position: "absolute",
    top: "-60px", right: "-60px",
    width: "220px", height: "220px",
    borderRadius: "50%",
    background: "rgba(52,211,153,0.12)",
    pointerEvents: "none"
  },
  headerTitle: {
    fontSize: "32px",
    fontWeight: "800",
    letterSpacing: "-0.5px",
    margin: 0
  },
  headerSub: {
    margin: "6px 0 0",
    color: "rgba(255,255,255,0.65)",
    fontSize: "14px",
    letterSpacing: "0.4px"
  },
  tabBar: {
    display: "flex",
    borderBottom: "1px solid #e8ecf0",
    background: "#f8fafc",
    overflowX: "auto"
  },
  tab: (active, color) => ({
    flex: 1,
    padding: "16px 10px",
    border: "none",
    background: active ? "white" : "transparent",
    borderBottom: active ? `3px solid ${color}` : "3px solid transparent",
    color: active ? color : "#94a3b8",
    fontWeight: "700",
    fontSize: "12px",
    cursor: "pointer",
    letterSpacing: "0.3px",
    transition: "all 0.2s",
    whiteSpace: "nowrap"
  }),
  body: { padding: "36px 40px" },
  grid2: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "40px",
    alignItems: "start"
  },
  sectionTitle: (color) => ({
    fontSize: "15px",
    fontWeight: "700",
    color,
    borderBottom: `2px solid ${color}`,
    paddingBottom: "10px",
    marginBottom: "22px",
    letterSpacing: "0.3px"
  }),
  inputGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "14px"
  },
  label: {
    fontSize: "11px",
    fontWeight: "700",
    color: "#64748b",
    letterSpacing: "0.8px",
    textTransform: "uppercase"
  },
  unitHint: {
    fontSize: "10px",
    color: "#94a3b8",
    marginLeft: "6px",
    fontWeight: "400"
  },
  input: (hasError) => ({
    width: "100%",
    padding: "11px 12px",
    marginTop: "5px",
    borderRadius: "10px",
    border: `1.5px solid ${hasError ? "#ef4444" : "#e2e8f0"}`,
    fontSize: "14px",
    outline: "none",
    background: hasError ? "#fff5f5" : "white",
    boxSizing: "border-box",
    transition: "border-color 0.2s"
  }),
  btnPrimary: (disabled, color = "#0ea5e9") => ({
    width: "100%",
    padding: "14px",
    marginTop: "20px",
    background: disabled
      ? "#cbd5e1"
      : `linear-gradient(135deg, ${color}, ${color}bb)`,
    color: "white",
    border: "none",
    borderRadius: "12px",
    cursor: disabled ? "not-allowed" : "pointer",
    fontWeight: "700",
    fontSize: "15px",
    letterSpacing: "0.3px",
    transition: "opacity 0.2s"
  }),
  btnSecondary: {
    width: "100%",
    marginTop: "10px",
    padding: "11px",
    background: "white",
    color: "#ef4444",
    border: "1.5px solid #ef4444",
    borderRadius: "12px",
    cursor: "pointer",
    fontWeight: "600",
    fontSize: "14px"
  },
  errorBanner: {
    background: "#fff1f2",
    border: "1px solid #fecaca",
    borderRadius: "10px",
    padding: "12px 16px",
    color: "#dc2626",
    fontSize: "13px",
    marginTop: "14px",
    display: "flex",
    alignItems: "center",
    gap: "8px"
  },
  infoBanner: {
    background: "#fffbeb",
    border: "1px solid #fde68a",
    borderRadius: "10px",
    padding: "12px 16px",
    color: "#92400e",
    fontSize: "13px",
    marginBottom: "24px",
    lineHeight: "1.5"
  },
  resultBox: (category) => ({
    textAlign: "center",
    background: CATEGORY_CONFIG[category]?.bg || "#f8fafc",
    borderRadius: "16px",
    padding: "28px 20px",
    border: `1.5px solid ${CATEGORY_CONFIG[category]?.color || "#e2e8f0"}22`
  }),
  aqiNumber: (category) => ({
    fontSize: "72px",
    fontWeight: "800",
    color: CATEGORY_CONFIG[category]?.color || "#334155",
    lineHeight: 1,
    margin: "8px 0"
  }),
  badge: (category) => ({
    display: "inline-block",
    padding: "8px 24px",
    borderRadius: "30px",
    background: CATEGORY_CONFIG[category]?.color || "#334155",
    color: "white",
    fontWeight: "700",
    fontSize: "15px",
    letterSpacing: "0.4px",
    margin: "10px 0"
  }),
  adviceBox: {
    background: "white",
    borderRadius: "12px",
    padding: "14px 16px",
    marginTop: "16px",
    fontSize: "13px",
    color: "#475569",
    lineHeight: "1.6",
    textAlign: "left",
    boxShadow: "0 1px 4px rgba(0,0,0,0.06)"
  },
  confidenceRow: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    marginBottom: "8px"
  },
  confidenceBar: (width, color) => ({
    height: "6px",
    borderRadius: "4px",
    background: color,
    width: `${width}%`,
    transition: "width 0.5s ease",
    minWidth: "2px"
  }),
  metricsTable: {
    width: "100%",
    borderCollapse: "collapse",
    marginTop: "20px",
    fontSize: "13px"
  },
  th: {
    background: "#0f2027",
    color: "white",
    padding: "12px 14px",
    textAlign: "left",
    fontWeight: "600",
    letterSpacing: "0.3px"
  },
  td: (i) => ({
    padding: "11px 14px",
    background: i % 2 === 0 ? "#f8fafc" : "white",
    borderBottom: "1px solid #f1f5f9",
    color: "#334155"
  }),
  statCard: (color) => ({
    background: `${color}12`,
    border: `1px solid ${color}30`,
    borderRadius: "14px",
    padding: "20px",
    textAlign: "center"
  }),
  statValue: (color) => ({
    fontSize: "28px",
    fontWeight: "800",
    color
  }),
  statLabel: {
    fontSize: "12px",
    color: "#64748b",
    marginTop: "4px",
    fontWeight: "600",
    letterSpacing: "0.4px",
    textTransform: "uppercase"
  },
  placeholder: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "260px",
    color: "#94a3b8",
    gap: "12px"
  },
  forecastCard: (category) => ({
    background: CATEGORY_CONFIG[category]?.bg || "#f8fafc",
    border: `1.5px solid ${CATEGORY_CONFIG[category]?.color || "#e2e8f0"}44`,
    borderRadius: "14px",
    padding: "14px 8px",
    textAlign: "center",
    flex: "1 1 0"
  }),
  forecastAQI: (category) => ({
    fontSize: "24px",
    fontWeight: "800",
    color: CATEGORY_CONFIG[category]?.color || "#334155",
    lineHeight: 1,
    margin: "5px 0"
  })
};

// ─── Main Component ───────────────────────────────────────────────────────────
export default function App() {

  // ── Predictor state ──────────────────────────────────────────────────────
  const [activeTab, setActiveTab]     = useState("predictor");
  const [formData, setFormData]       = useState(EMPTY_FORM);
  const [fieldErrors, setFieldErrors] = useState({});
  const [result, setResult]           = useState(null);
  const [error, setError]             = useState(null);
  const [loading, setLoading]         = useState(false);

  // ── Analytics state ──────────────────────────────────────────────────────
  const [metrics, setMetrics]       = useState([]);
  const [featureData, setFeatureData] = useState([]);
  const [regMetrics, setRegMetrics]   = useState(null);

  // ── Forecast state ───────────────────────────────────────────────────────
  const [recentAQI, setRecentAQI]             = useState(["", "", ""]);
  const [forecastResult, setForecastResult]   = useState(null);
  const [forecastError, setForecastError]     = useState(null);
  const [forecastLoading, setForecastLoading] = useState(false);

  // ── Fetch analytics data on mount ────────────────────────────────────────
  const fetchDashboardData = useCallback(async () => {
    try {
      const [mRes, fRes, rRes] = await Promise.all([
        axios.get(`${API}/model-metrics`),
        axios.get(`${API}/feature-importance`),
        axios.get(`${API}/regression-metrics`)
      ]);
      setMetrics(mRes.data);
      setFeatureData(fRes.data);
      setRegMetrics(rRes.data);
    } catch (err) {
      console.error("Dashboard data fetch error:", err);
    }
  }, []);

  useEffect(() => { fetchDashboardData(); }, [fetchDashboardData]);

  // ── Predictor handlers ────────────────────────────────────────────────────
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    if (fieldErrors[name]) setFieldErrors(prev => ({ ...prev, [name]: false }));
    setError(null);
  };

  const handleReset = () => {
    setFormData(EMPTY_FORM);
    setResult(null);
    setError(null);
    setFieldErrors({});
  };

  const validateForm = () => {
    const errors = {};
    Object.keys(formData).forEach(key => {
      const val = parseFloat(formData[key]);
      const hi  = parseInt(POLLUTANT_META[key]?.hint.split("–")[1] || "9999");
      if (formData[key] === "" || isNaN(val)) errors[key] = true;
      else if (val < 0 || val > hi)           errors[key] = true;
    });
    return errors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    const errors = validateForm();
    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      setError("Please fix the highlighted fields before submitting.");
      return;
    }
    setLoading(true);
    try {
      const response = await axios.post(`${API}/predict`, formData);
      setResult(response.data);
    } catch (err) {
      const msg = err.response?.data?.error
        || "Could not reach the backend. Make sure Flask is running.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  // ── Forecast handlers ─────────────────────────────────────────────────────
  const handleRecentAQIChange = (index, value) => {
    const updated = [...recentAQI];
    updated[index] = value;
    setRecentAQI(updated);
    setForecastError(null);
  };

  const validateForecastInputs = () => {
    // Pollutant fields
    const pollErr = validateForm();
    if (Object.keys(pollErr).length > 0) {
      setFieldErrors(pollErr);
      return "Please fill in all pollutant values correctly.";
    }
    // Recent AQI history
    const labels = ["3 days ago", "2 days ago", "Yesterday"];
    for (let i = 0; i < 3; i++) {
      const v = parseFloat(recentAQI[i]);
      if (recentAQI[i] === "" || isNaN(v) || v < 0 || v > 500) {
        return `Please enter a valid AQI (0–500) for "${labels[i]}".`;
      }
    }
    return null;
  };

  const handleForecast = async (e) => {
    e.preventDefault();
    setForecastError(null);
    setForecastResult(null);

    const validErr = validateForecastInputs();
    if (validErr) { setForecastError(validErr); return; }

    setForecastLoading(true);
    try {
      const response = await axios.post(`${API}/forecast`, {
        ...formData,
        recent_aqi: recentAQI.map(Number)
      });
      setForecastResult(response.data.predictions);
    } catch (err) {
      const msg = err.response?.data?.error
        || "Forecast failed. Make sure Flask is running and forecast models are trained (run train_forecast.py).";
      setForecastError(msg);
    } finally {
      setForecastLoading(false);
    }
  };

  const handleForecastReset = () => {
    setRecentAQI(["", "", ""]);
    setForecastResult(null);
    setForecastError(null);
    setFieldErrors({});
    setFormData(EMPTY_FORM);
  };

  // ── Chart data ─────────────────────────────────────────────────────────────
  const accuracyChartData = {
    labels: metrics.map(m => m.model),
    datasets: [
      {
        label: "Accuracy",
        data: metrics.map(m => +(m.accuracy * 100).toFixed(2)),
        backgroundColor: "rgba(14,165,233,0.75)",
        borderRadius: 6
      },
      {
        label: "F1 Score",
        data: metrics.map(m => +(m.f1_score * 100).toFixed(2)),
        backgroundColor: "rgba(16,185,129,0.75)",
        borderRadius: 6
      }
    ]
  };

  const featureChartData = {
    labels: featureData.map(f => f.Feature),
    datasets: [
      {
        label: "Importance",
        data: featureData.map(f => +f.Importance.toFixed(4)),
        backgroundColor: featureData.map((_, i) =>
          `hsl(${160 + i * 22}, 65%, 48%)`
        ),
        borderRadius: 6
      }
    ]
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: "top" },
      tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y}%` } }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: { display: true, text: "Score (%)", color: "#64748b" }
      }
    }
  };

  // Forecast line chart
  const forecastChartData = forecastResult ? {
    labels: forecastResult.map(p => p.label),
    datasets: [
      {
        label: "Predicted AQI",
        data: forecastResult.map(p => p.aqi),
        borderColor: "#f59e0b",
        backgroundColor: "rgba(245,158,11,0.10)",
        pointBackgroundColor: forecastResult.map(
          p => CATEGORY_CONFIG[p.category]?.color || "#f59e0b"
        ),
        pointBorderColor: "white",
        pointBorderWidth: 2,
        pointRadius: 8,
        pointHoverRadius: 10,
        borderWidth: 2.5,
        tension: 0.35,
        fill: true
      }
    ]
  } : null;

  const forecastChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: ctx => ` AQI: ${ctx.parsed.y}`,
          afterLabel: ctx => {
            const p = forecastResult[ctx.dataIndex];
            return ` ${p.category} ${CATEGORY_CONFIG[p.category]?.icon || ""}`;
          }
        }
      }
    },
    scales: {
      y: {
        min: 0, max: 500,
        title: { display: true, text: "AQI Value", color: "#64748b" },
        grid: { color: "#f1f5f9" }
      },
      x: { grid: { display: false } }
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={styles.root}>
      <div style={styles.card}>

        {/* ── Header ──────────────────────────────────────────────────── */}
        <div style={styles.header}>
          <div style={styles.headerGlow} />
          <h1 style={styles.headerTitle}>🌍 EcoVision</h1>
          <p style={styles.headerSub}>
            AI-Powered Environmental Pollution Prediction & Analysis System
          </p>
        </div>

        {/* ── Tab Bar ─────────────────────────────────────────────────── */}
        <div style={styles.tabBar}>
          {[
            { key: "predictor", label: "🔍 AQI Predictor",     color: "#0ea5e9" },
            { key: "forecast",  label: "📈 AQI Forecast",      color: "#f59e0b" },
            { key: "analytics", label: "📊 Model Analytics",   color: "#8b5cf6" },
            { key: "insights",  label: "🧪 Pollutant Insights", color: "#10b981" }
          ].map(t => (
            <button
              key={t.key}
              onClick={() => setActiveTab(t.key)}
              style={styles.tab(activeTab === t.key, t.color)}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div style={styles.body}>

          {/* ══════════════════════════════════════════════════════════
              TAB 1 — AQI Predictor
          ══════════════════════════════════════════════════════════ */}
          {activeTab === "predictor" && (
            <div style={styles.grid2}>

              {/* Left — Input Form */}
              <section>
                <p style={styles.sectionTitle("#0ea5e9")}>Input Pollutant Readings</p>
                <form onSubmit={handleSubmit}>
                  <div style={styles.inputGrid}>
                    {Object.keys(formData).map(key => {
                      const meta = POLLUTANT_META[key];
                      return (
                        <div key={key}>
                          <label style={styles.label}>
                            {key}
                            <span style={styles.unitHint}>
                              {meta.unit} · {meta.hint}
                            </span>
                          </label>
                          <input
                            type="number"
                            step="any"
                            name={key}
                            value={formData[key]}
                            onChange={handleChange}
                            placeholder={meta.placeholder}
                            style={styles.input(!!fieldErrors[key])}
                          />
                        </div>
                      );
                    })}
                  </div>

                  {error && (
                    <div style={styles.errorBanner}>⚠️ {error}</div>
                  )}

                  <button type="submit" disabled={loading} style={styles.btnPrimary(loading)}>
                    {loading ? "⏳ Analyzing..." : "🔬 Analyze Air Quality"}
                  </button>
                  <button type="button" onClick={handleReset} style={styles.btnSecondary}>
                    ✕ Clear Fields
                  </button>
                </form>
              </section>

              {/* Right — Result */}
              <section>
                <p style={styles.sectionTitle("#10b981")}>Prediction Result</p>

                {result ? (
                  <div style={styles.resultBox(result.predicted_category)}>
                    <GaugeChart
                      id="aqi-gauge"
                      nrOfLevels={6}
                      arcWidth={0.28}
                      colors={["#00c853","#76c442","#ffc107","#ff6f00","#e53935","#7b1fa2"]}
                      percent={Math.min(result.predicted_aqi / 500, 1)}
                      hideText={true}
                      animate={true}
                      style={{ width: "85%", margin: "0 auto" }}
                    />

                    <div style={styles.aqiNumber(result.predicted_category)}>
                      {result.predicted_aqi}
                    </div>

                    <div>
                      <span style={styles.badge(result.predicted_category)}>
                        {CATEGORY_CONFIG[result.predicted_category]?.icon}{" "}
                        {result.predicted_category}
                      </span>
                    </div>

                    {/* Health Advice */}
                    <div style={styles.adviceBox}>
                      <strong style={{ color: "#0f172a", fontSize: "12px", letterSpacing: "0.4px" }}>
                        🏥 HEALTH ADVISORY
                      </strong>
                      <p style={{ margin: "6px 0 0" }}>{result.health_advice}</p>
                    </div>

                    {/* Confidence Scores */}
                    {result.confidence_scores &&
                      Object.keys(result.confidence_scores).length > 0 && (
                      <div style={{ marginTop: "20px", textAlign: "left" }}>
                        <p style={{
                          fontSize: "11px", fontWeight: "700", color: "#64748b",
                          marginBottom: "10px", letterSpacing: "0.6px",
                          textTransform: "uppercase"
                        }}>
                          Model Confidence
                        </p>
                        {Object.entries(result.confidence_scores)
                          .sort((a, b) => b[1] - a[1])
                          .map(([cat, prob]) => (
                            <div key={cat} style={styles.confidenceRow}>
                              <span style={{ fontSize: "11px", width: "90px", color: "#475569", fontWeight: "600" }}>
                                {cat}
                              </span>
                              <div style={{ flex: 1, background: "#f1f5f9", borderRadius: "4px", height: "6px" }}>
                                <div style={styles.confidenceBar(
                                  prob * 100,
                                  CATEGORY_CONFIG[cat]?.color || "#94a3b8"
                                )} />
                              </div>
                              <span style={{ fontSize: "11px", color: "#64748b", width: "40px", textAlign: "right" }}>
                                {(prob * 100).toFixed(1)}%
                              </span>
                            </div>
                          ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={styles.placeholder}>
                    <span style={{ fontSize: "52px" }}>🌫</span>
                    <p style={{ fontSize: "14px" }}>Enter pollutant values and click Analyze</p>
                  </div>
                )}
              </section>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════
              TAB 2 — AQI Forecast (NEW)
          ══════════════════════════════════════════════════════════ */}
          {activeTab === "forecast" && (
            <>
              <p style={styles.sectionTitle("#f59e0b")}>📈 AQI Forecast — Next 7 Days</p>

              <div style={styles.infoBanner}>
                ℹ️ Enter <strong>today's pollutant readings</strong> and the <strong>AQI values from
                the last 3 days</strong>. EcoVision will predict air quality for the next 7 days
                using machine learning with temporal lag features.
              </div>

              <div style={styles.grid2}>

                {/* Left — Inputs */}
                <section>
                  {/* Pollutant inputs */}
                  <p style={{
                    fontSize: "11px", fontWeight: "700", color: "#64748b",
                    marginBottom: "12px", letterSpacing: "0.8px",
                    textTransform: "uppercase"
                  }}>
                    Today's Pollutant Readings
                  </p>

                  <div style={styles.inputGrid}>
                    {Object.keys(formData).map(key => {
                      const meta = POLLUTANT_META[key];
                      return (
                        <div key={key}>
                          <label style={styles.label}>
                            {key}
                            <span style={styles.unitHint}>
                              {meta.unit} · {meta.hint}
                            </span>
                          </label>
                          <input
                            type="number"
                            step="any"
                            name={key}
                            value={formData[key]}
                            onChange={handleChange}
                            placeholder={meta.placeholder}
                            style={styles.input(!!fieldErrors[key])}
                          />
                        </div>
                      );
                    })}
                  </div>

                  {/* Recent AQI History */}
                  <p style={{
                    fontSize: "11px", fontWeight: "700", color: "#64748b",
                    margin: "22px 0 12px", letterSpacing: "0.8px",
                    textTransform: "uppercase"
                  }}>
                    Recent AQI History (0 – 500)
                  </p>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px" }}>
                    {["3 Days Ago", "2 Days Ago", "Yesterday"].map((label, i) => (
                      <div key={i}>
                        <label style={styles.label}>{label}</label>
                        <input
                          type="number"
                          step="any"
                          min="0"
                          max="500"
                          value={recentAQI[i]}
                          onChange={e => handleRecentAQIChange(i, e.target.value)}
                          placeholder="e.g. 180"
                          style={styles.input(false)}
                        />
                      </div>
                    ))}
                  </div>

                  {forecastError && (
                    <div style={styles.errorBanner}>⚠️ {forecastError}</div>
                  )}

                  <button
                    onClick={handleForecast}
                    disabled={forecastLoading}
                    style={styles.btnPrimary(forecastLoading, "#f59e0b")}
                  >
                    {forecastLoading ? "⏳ Forecasting..." : "📈 Generate 7-Day Forecast"}
                  </button>

                  <button type="button" onClick={handleForecastReset} style={styles.btnSecondary}>
                    ✕ Reset All
                  </button>
                </section>

                {/* Right — Forecast Result */}
                <section>
                  <p style={styles.sectionTitle("#f59e0b")}>Forecast Result</p>

                  {forecastResult ? (
                    <>
                      {/* Line Chart */}
                      <div style={{ height: "220px", marginBottom: "20px" }}>
                        <Line data={forecastChartData} options={forecastChartOptions} />
                      </div>

                      {/* Day Cards */}
                      <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", marginBottom: "16px" }}>
                        {forecastResult.map((p, i) => (
                          <div key={i} style={styles.forecastCard(p.category)}>
                            <div style={{
                              fontSize: "9px", fontWeight: "700",
                              color: "#64748b", marginBottom: "3px",
                              letterSpacing: "0.4px", textTransform: "uppercase"
                            }}>
                              {p.label}
                            </div>
                            <div style={styles.forecastAQI(p.category)}>
                              {p.aqi}
                            </div>
                            <div style={{ fontSize: "15px", margin: "3px 0" }}>
                              {CATEGORY_CONFIG[p.category]?.icon}
                            </div>
                            <div style={{
                              fontSize: "8px", fontWeight: "700",
                              color: CATEGORY_CONFIG[p.category]?.color,
                              letterSpacing: "0.2px"
                            }}>
                              {p.category}
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Worst day warning */}
                      {(() => {
                        const worst = forecastResult.reduce(
                          (a, b) => a.aqi > b.aqi ? a : b
                        );
                        const best = forecastResult.reduce(
                          (a, b) => a.aqi < b.aqi ? a : b
                        );
                        return (
                          <>
                            <div style={{
                              background: `${CATEGORY_CONFIG[worst.category]?.color}15`,
                              border: `1px solid ${CATEGORY_CONFIG[worst.category]?.color}44`,
                              borderRadius: "10px",
                              padding: "12px 14px",
                              fontSize: "12px",
                              color: "#334155",
                              marginBottom: "8px"
                            }}>
                              <strong>⚠️ Peak Pollution — {worst.label}:</strong>{" "}
                              AQI {worst.aqi} ({worst.category}).{" "}
                              {HEALTH_ADVICE[worst.category]}
                            </div>
                            <div style={{
                              background: `${CATEGORY_CONFIG[best.category]?.color}15`,
                              border: `1px solid ${CATEGORY_CONFIG[best.category]?.color}44`,
                              borderRadius: "10px",
                              padding: "12px 14px",
                              fontSize: "12px",
                              color: "#334155"
                            }}>
                              <strong>✅ Best Day — {best.label}:</strong>{" "}
                              AQI {best.aqi} ({best.category}).{" "}
                              {HEALTH_ADVICE[best.category]}
                            </div>
                          </>
                        );
                      })()}
                    </>
                  ) : (
                    <div style={styles.placeholder}>
                      <span style={{ fontSize: "52px" }}>📅</span>
                      <p style={{ fontSize: "14px", textAlign: "center" }}>
                        Fill in the form and click<br />"Generate 7-Day Forecast"
                      </p>
                    </div>
                  )}
                </section>
              </div>
            </>
          )}

          {/* ══════════════════════════════════════════════════════════
              TAB 3 — Model Analytics
          ══════════════════════════════════════════════════════════ */}
          {activeTab === "analytics" && (
            <>
              <p style={styles.sectionTitle("#8b5cf6")}>Model Performance Analytics</p>

              {/* Regression stat cards */}
              {regMetrics && (
                <div style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(3,1fr)",
                  gap: "16px",
                  marginBottom: "36px"
                }}>
                  {[
                    { label: "R² Score", value: regMetrics.r2_score?.toFixed(4), color: "#8b5cf6" },
                    { label: "MAE",      value: regMetrics.mae?.toFixed(2),       color: "#0ea5e9" },
                    { label: "RMSE",     value: regMetrics.rmse?.toFixed(2),      color: "#10b981" }
                  ].map(s => (
                    <div key={s.label} style={styles.statCard(s.color)}>
                      <div style={styles.statValue(s.color)}>{s.value}</div>
                      <div style={styles.statLabel}>{s.label}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Bar chart */}
              <div style={{ height: "300px" }}>
                <Bar data={accuracyChartData} options={barChartOptions} />
              </div>

              {/* Metrics table */}
              {metrics.length > 0 && (
                <div style={{ marginTop: "36px", overflowX: "auto" }}>
                  <p style={styles.sectionTitle("#8b5cf6")}>Detailed Metrics Table</p>
                  <table style={styles.metricsTable}>
                    <thead>
                      <tr>
                        {["Model","Accuracy","Precision","Recall","F1 Score","CV F1 Mean","CV F1 Std"].map(h => (
                          <th key={h} style={styles.th}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.map((m, i) => (
                        <tr key={m.model}>
                          <td style={styles.td(i)}><strong>{m.model}</strong></td>
                          <td style={styles.td(i)}>{(m.accuracy  * 100).toFixed(2)}%</td>
                          <td style={styles.td(i)}>{(m.precision * 100).toFixed(2)}%</td>
                          <td style={styles.td(i)}>{(m.recall    * 100).toFixed(2)}%</td>
                          <td style={styles.td(i)}>{(m.f1_score  * 100).toFixed(2)}%</td>
                          <td style={styles.td(i)}>
                            {m.cv_f1_mean != null
                              ? (m.cv_f1_mean * 100).toFixed(2) + "%" : "—"}
                          </td>
                          <td style={styles.td(i)}>
                            {m.cv_f1_std != null
                              ? "± " + (m.cv_f1_std * 100).toFixed(2) + "%" : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}

          {/* ══════════════════════════════════════════════════════════
              TAB 4 — Pollutant Insights
          ══════════════════════════════════════════════════════════ */}
          {activeTab === "insights" && (
            <>
              <p style={styles.sectionTitle("#10b981")}>Pollutant Impact Analysis</p>

              <div style={{ height: "320px" }}>
                <Bar
                  data={featureChartData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: "y",
                    plugins: {
                      legend: { display: false },
                      tooltip: {
                        callbacks: { label: ctx => ` Importance: ${ctx.parsed.x}` }
                      }
                    },
                    scales: {
                      x: {
                        beginAtZero: true,
                        title: { display: true, text: "Feature Importance Score" }
                      }
                    }
                  }}
                />
              </div>

              {/* AQI Reference Table */}
              <div style={{ marginTop: "36px" }}>
                <p style={styles.sectionTitle("#10b981")}>AQI Category Reference</p>
                <table style={styles.metricsTable}>
                  <thead>
                    <tr>
                      {["Category","AQI Range","Health Impact","Recommended Action"].map(h => (
                        <th key={h} style={styles.th}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ["Good",         "0 – 50",    "Minimal impact",                  "Enjoy outdoor activities"],
                      ["Satisfactory", "51 – 100",  "Minor breathing discomfort",       "Sensitive groups take care"],
                      ["Moderate",     "101 – 200", "Discomfort for sensitive groups",  "Limit prolonged outdoor exertion"],
                      ["Poor",         "201 – 300", "Breathing discomfort for all",     "Wear mask outdoors"],
                      ["Very Poor",    "301 – 400", "Respiratory illness possible",     "Avoid outdoor activity"],
                      ["Severe",       "401+",      "Serious health hazard",            "Stay indoors, use air purifier"]
                    ].map(([cat, range, impact, action], i) => (
                      <tr key={cat}>
                        <td style={styles.td(i)}>
                          <span style={{
                            display: "inline-block",
                            padding: "3px 12px",
                            borderRadius: "20px",
                            background: CATEGORY_CONFIG[cat]?.color || "#94a3b8",
                            color: "white",
                            fontSize: "12px",
                            fontWeight: "700"
                          }}>
                            {CATEGORY_CONFIG[cat]?.icon} {cat}
                          </span>
                        </td>
                        <td style={styles.td(i)}>{range}</td>
                        <td style={styles.td(i)}>{impact}</td>
                        <td style={styles.td(i)}>{action}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

        </div>
      </div>
    </div>
  );
}