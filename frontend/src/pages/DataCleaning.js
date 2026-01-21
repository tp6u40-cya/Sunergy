// src/pages/DataCleaning.js
import React, { useEffect, useState } from "react";
import Navbar from "../components/Navbar";

/* Chart.js core */
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";

/* Boxplot 插件 */
import { BoxPlotController, BoxAndWiskers } from "@sgratzl/chartjs-chart-boxplot";

import { Scatter, Bar, Chart } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  BarElement,
  Tooltip,
  Legend,
  BoxPlotController,
  BoxAndWiskers
);

/* 卡片元件（點擊放大） */
const ChartCard = ({ title, children, onClick }) => (
  <div
    className="rounded-xl border border-white/10 bg-white/[.02] p-4 cursor-pointer hover:border-primary/50 transition-all"
    onClick={onClick}
  >
    <div className="text-sm text-white/70 mb-2">{title}</div>
    <div className="h-56">{children}</div>
  </div>
);

/* ScatterChart（實心點，TM vs GI / GI vs TM 不顯示離群值） */
const ScatterChart = ({ x, y, xLabel, yLabel, outlierMask, title }) => {
  if (!x || !y || !outlierMask) return <div className="text-xs text-white/40">無資料</div>;

  const isTMGI = title === "TM vs GI" || title === "GI vs TM";

  const normal = [];
  const outlier = [];

  x.forEach((vx, i) => {
    if (y[i] == null) return;
    const p = { x: vx, y: y[i], idx: i };
    if (!isTMGI && outlierMask[i]) outlier.push(p);
    else normal.push(p);
  });

  const datasets = [
    {
      label: "正常值",
      data: normal,
      backgroundColor: "#60a5fa",
      pointRadius: 4,
      pointHoverRadius: 7,
    },
  ];

  if (!isTMGI && outlier.length > 0) {
    datasets.push({
      label: "離群值",
      data: outlier,
      backgroundColor: "#ef4444",
      pointRadius: 6,
      pointHoverRadius: 9,
    });
  }

  return (
    <Scatter
      data={{ datasets }}
      options={{
        maintainAspectRatio: false,
        responsive: true,
        plugins: {
          legend: { display: true, position: "top", labels: { color: "#ccc", usePointStyle: true, padding: 15 } },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const p = ctx.raw;
                return [
                  `第 ${p.idx + 1} 筆`,
                  `${xLabel}: ${p.x.toFixed(2)}`,
                  `${yLabel}: ${p.y.toFixed(2)}`,
                  (!isTMGI && outlierMask[p.idx]) ? "⚠️ 離群值" : "正常值",
                ];
              },
            },
          },
        },
        scales: {
          x: { title: { display: true, text: xLabel, color: "#ccc" }, ticks: { color: "#ccc" } },
          y: { title: { display: true, text: yLabel, color: "#ccc" }, ticks: { color: "#ccc" } },
        },
      }}
    />
  );
};

/* Histogram（單色藍） */
const Histogram = ({ values }) => {
  if (!values || values.length === 0) return <div className="text-xs text-white/40">無資料</div>;

  const bins = 15;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const step = (max - min) / bins || 1;

  const counts = Array(bins).fill(0);
  values.forEach((v) => {
    const i = Math.min(Math.floor((v - min) / step), bins - 1);
    counts[i]++;
  });

  return (
    <Bar
      data={{
        labels: counts.map((_, i) => `${(min + i * step).toFixed(1)}`),
        datasets: [{ label: "筆數", data: counts, backgroundColor: "#60a5fa", borderColor: "#3b82f6", borderWidth: 1 }],
      }}
      options={{
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, ticks: { color: "#ccc" } }, x: { ticks: { color: "#ccc" } } },
      }}
    />
  );
};

/* BoxPlotChart（優化 Day of Year & Hour） */
const BoxPlotChart = ({ groupedData, title }) => {
  const isHour = title.includes("Hour");

  let processedData = groupedData || {};
  if (isHour) {
    processedData = {};
    for (let h = 0; h < 24; h++) {
      const key = String(h);
      processedData[key] = groupedData[key] || { values: [] };
    }
  }

  const entries = Object.entries(processedData).sort(([a], [b]) => Number(a) - Number(b));

  if (entries.length === 0) return <div className="text-xs text-white/40 text-center pt-8">無資料</div>;

  const labels = entries.map(([k]) => {
    if (title.includes("Month")) return `月份 ${k}`;
    if (title.includes("Day")) return `第 ${k} 天`;
    if (title.includes("Hour")) return `${k.padStart(2, "0")}:00`;
    return k;
  });

  const data = entries.map(([_, v]) => v.values || []);
  const totalPoints = data.reduce((sum, arr) => sum + arr.length, 0);

  const isDayOfYear = title.includes("Day of Year");

  return (
    <div className="relative h-full">
      <Chart
        type="boxplot"
        data={{
          labels,
          datasets: [
            {
              label: "EAC",
              data,
              backgroundColor: "rgba(96, 165, 250, 0.4)",
              borderColor: "#60a5fa",
              outlierBackgroundColor: "#ef4444",
              outlierBorderColor: "#dc2626",
              outlierRadius: 4,
              padding: isHour ? 20 : 10,
              itemRadius: 0,
              boxWidth: isHour ? 0.9 : (isDayOfYear ? 0.6 : 0.7),
              whiskerWidth: isHour ? 0.8 : 0.5,
            },
          ],
        }}
        options={{
          maintainAspectRatio: false,
          responsive: true,
          animation: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const values = ctx.raw || [];
                  if (values.length === 0) return "無資料";
                  return [`筆數: ${values.length}`];
                },
              },
            },
          },
          scales: {
            x: {
              ticks: {
                color: "#ccc",
                maxRotation: isDayOfYear ? 90 : 0,
                minRotation: isDayOfYear ? 90 : 0,
                autoSkip: isDayOfYear,
                maxTicksLimit: isDayOfYear ? 20 : undefined,
              },
            },
            y: { ticks: { color: "#ccc" }, title: { display: true, text: "EAC", color: "#ccc" }, beginAtZero: true },
          },
        }}
      />
      <div className="absolute top-2 right-2 text-[10px] text-white/40">
        {entries.length} 群組 / {totalPoints} 筆
      </div>
    </div>
  );
};

/* 主頁 */
export default function DataCleaning({ onBack, onNext, ...props }) {
  const [vizData, setVizData] = useState(null);
  const [removeOutliers, setRemoveOutliers] = useState(false);
  const [outlierMethod, setOutlierMethod] = useState("iqr");
  const [iqrFactor, setIqrFactor] = useState(1.5);
  const [zscoreThreshold, setZscoreThreshold] = useState(3.0);
  const [enlargedCard, setEnlargedCard] = useState(null);

  useEffect(() => {
    const id = localStorage.getItem("lastDataId");
    if (!id) return;

    const method = removeOutliers ? outlierMethod : "iqr";

    let url = `http://127.0.0.1:8000/visualize/site-data?data_id=${id}&remove_outliers=${removeOutliers}&outlier_method=${method}`;
    if (method === "iqr") url += `&iqr_factor=${iqrFactor}`;
    if (method === "zscore") url += `&zscore_threshold=${zscoreThreshold}`;

    fetch(url)
      .then((r) => r.json())
      .then(setVizData)
      .catch(console.error);
  }, [removeOutliers, outlierMethod, iqrFactor, zscoreThreshold]);

  if (!vizData) return <div className="text-white p-10">載入中...</div>;

  const pairs = vizData.scatter_matrix?.pairs || {};
  const outlierMask = vizData.outlier_mask || [];

  const handleCardClick = (title) => setEnlargedCard(enlargedCard === title ? null : title);

  const renderChartCard = (title, children) => (
    <ChartCard title={title} onClick={() => handleCardClick(title)}>
      {children}
    </ChartCard>
  );

  return (
    <div className="min-h-screen w-full bg-background-dark text-white flex flex-col">
      <Navbar activePage="predict" {...props} />

      <div className="w-full border-b border-white/10 bg-white/[.02] px-6 py-3 sticky top-[64px] sm:top-[65px] z-40 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <button onClick={onBack} className="flex items-center gap-1 text-sm text-white/50 hover:text-white transition-colors">
            <span className="material-symbols-outlined !text-lg">arrow_back</span>
            返回上一步
          </button>

          <div className="text-sm font-medium">
            <span className="text-white/40">1. 上傳資料</span>
            <span className="mx-2 text-white/30">/</span>
            <span className="text-primary font-bold">2. 清理資料</span>
            <span className="mx-2 text-white/30">/</span>
            <span className="text-white/40">3. 調整單位</span>
            <span className="mx-2 text-white/30">/</span>
            <span className="text-white/40">4. 選擇模型</span>
            <span className="mx-2 text-white/30">/</span>
            <span className="text-white/40">5. 輸出結果</span>
          </div>
        </div>
      </div>

      <main className="max-w-7xl mx-auto p-6">
        <h1 className="text-3xl font-bold mb-6">數據驗證與離群值處理</h1>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {renderChartCard("EAC 分佈", <Histogram values={pairs.GI__EAC?.y || []} />)}
          {renderChartCard("GI vs EAC", <ScatterChart title="GI vs EAC" x={pairs.GI__EAC?.x || []} y={pairs.GI__EAC?.y || []} xLabel="GI" yLabel="EAC" outlierMask={outlierMask} />)}
          {renderChartCard("TM vs EAC", <ScatterChart title="TM vs EAC" x={pairs.TM__EAC?.x || []} y={pairs.TM__EAC?.y || []} xLabel="TM" yLabel="EAC" outlierMask={outlierMask} />)}
          {renderChartCard("EAC vs GI", <ScatterChart title="EAC vs GI" x={pairs.GI__EAC?.y || []} y={pairs.GI__EAC?.x || []} xLabel="EAC" yLabel="GI" outlierMask={outlierMask} />)}
          {renderChartCard("GI 分佈", <Histogram values={pairs.GI__EAC?.x || []} />)}
          {renderChartCard("TM vs GI", <ScatterChart title="TM vs GI" x={pairs.GI__TM?.y || []} y={pairs.GI__TM?.x || []} xLabel="TM" yLabel="GI" outlierMask={outlierMask} />)}
          {renderChartCard("EAC vs TM", <ScatterChart title="EAC vs TM" x={pairs.TM__EAC?.y || []} y={pairs.TM__EAC?.x || []} xLabel="EAC" yLabel="TM" outlierMask={outlierMask} />)}
          {renderChartCard("GI vs TM", <ScatterChart title="GI vs TM" x={pairs.GI__TM?.x || []} y={pairs.GI__TM?.y || []} xLabel="GI" yLabel="TM" outlierMask={outlierMask} />)}
          {renderChartCard("TM 分佈", <Histogram values={pairs.TM__EAC?.x || []} />)}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-10">
          {renderChartCard("EAC by Month", <BoxPlotChart groupedData={vizData.boxplot_by_month || {}} title="EAC by Month" />)}
          {renderChartCard("EAC by Day of Year", <BoxPlotChart groupedData={vizData.boxplot_by_day || {}} title="EAC by Day of Year" />)}
          {renderChartCard("EAC by Hour", <BoxPlotChart groupedData={vizData.boxplot_by_hour || {}} title="EAC by Hour" />)}
        </div>
      </main>

      {/* ================= 底部控制列 ================= */}
      <div className="sticky bottom-0 z-40 w-full border-t border-white/10 bg-background-dark/90 backdrop-blur-lg px-6 py-4 shadow-2xl">
        <div className="mx-auto flex max-w-6xl items-center justify-between">

          {/* 左側：離群值資訊 */}
          <div className="flex items-center gap-6">
            <div className="flex flex-col">
              <span className="text-sm font-bold text-white">
                離群值處理選項
              </span>
              <span className="text-xs text-white/60">
                偵測到{" "}
                <span className="font-bold text-white">
                  {outlierMask.filter(Boolean).length}
                </span>{" "}
                個離群點，建議移除。
              </span>
            </div>

            {/* 偵測方法 */}
            <select
              value={outlierMethod}
              onChange={(e) => setOutlierMethod(e.target.value)}
              className="rounded border border-white/40 bg-black/40 px-2.5 py-1 text-xs text-white focus:outline-none focus:border-primary"
            >
              <option value="none">無</option>
              <option value="iqr">IQR</option>
              <option value="zscore">Z-Score</option>
              <option value="isolation_forest">Isolation Forest</option>
              <option value="custom">自訂</option>
            </select>

            {/* IQR factor */}
            {outlierMethod === "iqr" && (
              <div className="flex items-center gap-2">
                <span className="text-white/60">Factor</span>
                <input
                  type="number"
                  step="0.1"
                  min="0.5"
                  value={iqrFactor}
                  onChange={(e) => setIqrFactor(parseFloat(e.target.value) || 1.5)}
                  className="w-16 rounded border border-white/40 bg-black/40 px-2 py-1 text-xs text-white"
                />
              </div>
            )}

            {/* Z-score threshold */}
            {outlierMethod === "zscore" && (
              <div className="flex items-center gap-2">
                <span className="text-white/60">Thresh</span>
                <input
                  type="number"
                  step="0.5"
                  min="1"
                  value={zscoreThreshold}
                  onChange={(e) => setZscoreThreshold(parseFloat(e.target.value) || 3)}
                  className="w-16 rounded border border-white/40 bg-black/40 px-2 py-1 text-xs text-white"
                />
              </div>
            )}

            {/* 移除離群值 toggle */}
            <button
              onClick={() => setRemoveOutliers(!removeOutliers)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                removeOutliers ? "bg-primary" : "bg-white/20"
              }`}
            >
              <span
                className={`inline-block h-4 w-4 rounded-full bg-white transition-transform ${
                  removeOutliers ? "translate-x-6" : "translate-x-1"
                }`}
              />
            </button>
          </div>

          {/* 右側：流程按鈕 */}
          <div className="flex items-center gap-4">
            <button
              onClick={onBack}
              className="rounded-lg border border-white/30 px-6 py-2 text-sm font-bold text-white hover:bg-white/10 transition"
            >
              返回上一步
            </button>

            <button
              onClick={onNext}
              className="rounded-lg bg-primary px-8 py-2 text-sm font-bold text-background-dark hover:scale-105 transition-transform"
            >
              下一步
            </button>
          </div>
        </div>
      </div>

      {/* 放大 Modal */}
      {enlargedCard && (
        <div className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-8" onClick={() => setEnlargedCard(null)}>
          <div className="bg-background-dark/95 rounded-2xl p-10 max-w-6xl w-full max-h-full overflow-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex justify-between items-center mb-8">
              <h2 className="text-4xl font-bold text-white">{enlargedCard}</h2>
              <button onClick={() => setEnlargedCard(null)} className="text-white/70 hover:text-white text-5xl">×</button>
            </div>
            <div className="h-[80vh]">
              {enlargedCard === "EAC 分佈" && <Histogram values={pairs.GI__EAC?.y || []} />}
              {enlargedCard === "GI vs EAC" && <ScatterChart title={enlargedCard} x={pairs.GI__EAC?.x || []} y={pairs.GI__EAC?.y || []} xLabel="GI" yLabel="EAC" outlierMask={outlierMask} />}
              {enlargedCard === "TM vs EAC" && <ScatterChart title={enlargedCard} x={pairs.TM__EAC?.x || []} y={pairs.TM__EAC?.y || []} xLabel="TM" yLabel="EAC" outlierMask={outlierMask} />}
              {enlargedCard === "EAC vs GI" && <ScatterChart title={enlargedCard} x={pairs.GI__EAC?.y || []} y={pairs.GI__EAC?.x || []} xLabel="EAC" yLabel="GI" outlierMask={outlierMask} />}
              {enlargedCard === "GI 分佈" && <Histogram values={pairs.GI__EAC?.x || []} />}
              {enlargedCard === "TM vs GI" && <ScatterChart title={enlargedCard} x={pairs.GI__TM?.y || []} y={pairs.GI__TM?.x || []} xLabel="TM" yLabel="GI" outlierMask={outlierMask} />}
              {enlargedCard === "EAC vs TM" && <ScatterChart title={enlargedCard} x={pairs.TM__EAC?.y || []} y={pairs.TM__EAC?.x || []} xLabel="EAC" yLabel="TM" outlierMask={outlierMask} />}
              {enlargedCard === "GI vs TM" && <ScatterChart title={enlargedCard} x={pairs.GI__TM?.x || []} y={pairs.GI__TM?.y || []} xLabel="GI" yLabel="TM" outlierMask={outlierMask} />}
              {enlargedCard === "TM 分佈" && <Histogram values={pairs.TM__EAC?.x || []} />}
              {enlargedCard === "EAC by Month" && <BoxPlotChart groupedData={vizData.boxplot_by_month || {}} title="EAC by Month" />}
              {enlargedCard === "EAC by Day of Year" && <BoxPlotChart groupedData={vizData.boxplot_by_day || {}} title="EAC by Day of Year" />}
              {enlargedCard === "EAC by Hour" && <BoxPlotChart groupedData={vizData.boxplot_by_hour || {}} title="EAC by Hour" />}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}