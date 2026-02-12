// src/pages/PredictSolar.js
import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';

/* ── 誤差燈號組件 ── */
const ErrorLight = ({ pct }) => {
  if (pct === null || pct === undefined) return <span className="text-white/20 text-[10px]">—</span>;
  const v = Number(pct);
  if (v <= 5) return (
    <span className="inline-flex items-center gap-1.5">
      <span className="size-2.5 rounded-full bg-green-400 shadow-[0_0_6px_rgba(34,197,94,0.5)]" />
      <span className="text-green-400 text-[10px] font-mono">{v.toFixed(1)}%</span>
    </span>
  );
  if (v <= 15) return (
    <span className="inline-flex items-center gap-1.5">
      <span className="size-2.5 rounded-full bg-yellow-400 shadow-[0_0_6px_rgba(234,179,8,0.5)]" />
      <span className="text-yellow-400 text-[10px] font-mono">{v.toFixed(1)}%</span>
    </span>
  );
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="size-2.5 rounded-full bg-red-400 shadow-[0_0_6px_rgba(239,68,68,0.5)] animate-pulse" />
      <span className="text-red-400 text-[10px] font-mono">{v.toFixed(1)}%</span>
    </span>
  );
};

/* ── 整體診斷燈號 ── */
const OverallStatus = ({ avgError }) => {
  if (avgError === null || avgError === undefined) return null;
  const v = Number(avgError);
  let cfg = { color: 'text-green-400', bg: 'bg-green-500', shadow: 'shadow-[0_0_15px_rgba(34,197,94,0.4)]', label: '發電正常', desc: '預測與實際高度吻合' };
  if (v > 15) cfg = { color: 'text-red-400', bg: 'bg-red-500', shadow: 'shadow-[0_0_15px_rgba(239,68,68,0.4)]', label: '發電異常', desc: '偏差過大，請檢查設備' };
  else if (v > 5) cfg = { color: 'text-yellow-400', bg: 'bg-yellow-500', shadow: 'shadow-[0_0_15px_rgba(234,179,8,0.4)]', label: '需留意', desc: '環境干擾或輕微積塵' };

  return (
    <div className="bg-white/5 border border-white/10 p-5 rounded-2xl flex items-center gap-4">
      <div className="relative flex items-center justify-center">
        <div className={`size-10 rounded-full ${cfg.bg} ${cfg.shadow} animate-pulse`} />
      </div>
      <div className="flex flex-col">
        <p className="text-[10px] text-white/40 uppercase font-bold tracking-widest mb-0.5">系統診斷狀態</p>
        <span className={`text-lg font-black ${cfg.color}`}>{cfg.label}</span>
        <span className="text-[10px] text-white/30 italic">{cfg.desc}</span>
      </div>
    </div>
  );
};

export default function PredictSolar({ onBack, onNavigateToDashboard, onLogout, onNavigateToSites, onNavigateToTrain, onNavigateToPredict, onNavigateToModelMgmt }) {
  const [file, setFile] = useState(null);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [trainedModels, setTrainedModels] = useState([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // pagination
  const PAGE_SIZE = 20;
  const [page, setPage] = useState(0);

  // Fetch trained models on mount
  useEffect(() => {
    fetch('http://127.0.0.1:8000/train/trained-models')
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) setTrainedModels(data); })
      .catch(() => { });
  }, []);

  const handlePredict = async () => {
    if (!file || !selectedModelId) return alert('請上傳資料並選擇模型');
    setIsPredicting(true);
    setError('');
    setResult(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model_id', selectedModelId);

      const res = await fetch('http://127.0.0.1:8000/train/predict-file', {
        method: 'POST',
        body: formData,
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.detail || '預測失敗');
      setResult(json);
      setPage(0);
    } catch (e) {
      setError(e.message || '預測過程發生錯誤');
    } finally {
      setIsPredicting(false);
    }
  };

  const navProps = { onNavigateToDashboard, onNavigateToTrain, onNavigateToPredict, onNavigateToSites, onNavigateToModelMgmt, onLogout };

  // Pagination helpers
  const rows = result?.rows || [];
  const totalPages = Math.ceil(rows.length / PAGE_SIZE);
  const pagedRows = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  // Columns to display (original columns + predicted_EAC + error_pct)
  const displayCols = result ? result.columns : [];
  // Highlight these columns
  const highlightCols = new Set(['EAC', 'predicted_EAC', 'error_pct']);

  return (
    <div className="min-h-screen w-full bg-background-dark text-white flex flex-col font-sans">
      <Navbar activePage="predict-solar" {...navProps} />

      <main className="flex-1 w-full max-w-[1400px] mx-auto p-6 py-10 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">

        {/* ── 左側：配置區 ── */}
        <div className="lg:col-span-3 flex flex-col gap-6">
          <section className="bg-white/[0.02] p-6 rounded-2xl border border-white/10 shadow-2xl">
            <h2 className="text-lg font-bold text-white mb-6 flex items-center gap-3 italic">
              <div className="size-8 rounded-lg bg-primary/20 text-primary flex items-center justify-center">
                <span className="material-symbols-outlined !text-xl">settings_applications</span>
              </div>
              預測配置
            </h2>

            <div className="space-y-6">
              {/* 1. Upload */}
              <div>
                <label className="text-[11px] text-white/30 mb-2 block font-bold uppercase tracking-widest">1. 上傳預測資料</label>
                <div
                  className="group border-2 border-dashed border-white/10 rounded-2xl p-6 text-center hover:bg-white/[0.03] hover:border-primary/50 transition-all cursor-pointer"
                  onClick={() => document.getElementById('predictFileInput').click()}
                >
                  <input type="file" id="predictFileInput" hidden accept=".csv,.xlsx,.xls" onChange={(e) => setFile(e.target.files[0])} />
                  <div className="size-10 rounded-full bg-white/5 mx-auto mb-3 flex items-center justify-center group-hover:bg-primary/10 transition-colors">
                    <span className="material-symbols-outlined !text-xl text-white/30 group-hover:text-primary">upload_file</span>
                  </div>
                  <p className="text-xs font-bold text-white/50 group-hover:text-white transition-colors">{file ? file.name : 'CSV / XLSX 檔案'}</p>
                </div>
              </div>

              {/* 2. Model Select */}
              <div>
                <label className="text-[11px] text-white/30 mb-2 block font-bold uppercase tracking-widest">2. 選擇訓練模型</label>
                <select
                  value={selectedModelId}
                  onChange={(e) => setSelectedModelId(e.target.value)}
                  className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-xs text-white focus:ring-2 focus:ring-primary/50 outline-none appearance-none cursor-pointer"
                >
                  <option value="">請選擇模型...</option>
                  {trainedModels.map(m => (
                    <option key={m.model_id} value={m.model_id}>
                      #{m.model_id} {m.model_type} — {m.trained_at ? m.trained_at.slice(0, 16).replace('T', ' ') : ''}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <button
              onClick={handlePredict}
              disabled={isPredicting || !file || !selectedModelId}
              className="w-full bg-primary text-background-dark py-4 rounded-2xl font-black text-sm hover:scale-[1.02] active:scale-95 transition-all shadow-[0_10px_30px_rgba(242,204,13,0.2)] mt-6 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
            >
              {isPredicting ? '運算執行中...' : '開始執行預測'}
            </button>

            {error && (
              <div className="mt-4 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-xs">
                {error}
              </div>
            )}
          </section>

          {/* Summary cards — only show after result */}
          {result && (
            <section className="space-y-4 animate-fade-in">
              <div className="grid grid-cols-1 gap-4">
                <div className="bg-white/5 border border-white/10 p-5 rounded-2xl">
                  <p className="text-[10px] text-white/40 uppercase font-bold mb-2 tracking-widest">預估發電總量</p>
                  <div className="flex items-baseline gap-2">
                    <span className="text-2xl font-black font-mono text-white">{result.total_predicted_eac?.toLocaleString() ?? '—'}</span>
                    <span className="text-xs text-primary font-bold">kWh</span>
                  </div>
                </div>
                <div className="bg-white/5 border border-white/10 p-5 rounded-2xl">
                  <p className="text-[10px] text-white/40 uppercase font-bold mb-2 tracking-widest">平均誤差</p>
                  <div className="flex items-baseline gap-2">
                    <span className="text-2xl font-black font-mono text-white">{result.avg_error_pct !== null ? result.avg_error_pct.toFixed(2) : '—'}</span>
                    <span className="text-xs text-primary font-bold">%</span>
                  </div>
                </div>
                <OverallStatus avgError={result.avg_error_pct} />
              </div>
            </section>
          )}
        </div>

        {/* ── 右側：資料表格 ── */}
        <div className="lg:col-span-9 flex flex-col">
          <div className={`flex-1 w-full rounded-2xl border border-white/10 bg-white/[0.01] p-6 flex flex-col relative transition-all shadow-2xl ${!result && 'items-center justify-center border-dashed opacity-40 min-h-[500px]'}`}>
            {result ? (
              <div className="w-full flex flex-col animate-fade-in">
                {/* Header */}
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-sm font-bold text-white flex items-center gap-2">
                    <span className="material-symbols-outlined !text-base text-primary">table_chart</span>
                    預測結果 <span className="text-white/30 font-normal ml-2">共 {result.total_rows} 筆</span>
                  </h2>
                  <div className="text-[10px] text-white/30">
                    模型：<span className="text-primary font-bold">{result.model_type}</span>
                  </div>
                </div>

                {/* Table */}
                <div className="overflow-x-auto rounded-xl border border-white/5">
                  <table className="w-full text-[10px] text-left whitespace-nowrap">
                    <thead className="bg-white/5 text-white/40 uppercase sticky top-0 z-10">
                      <tr>
                        <th className="px-3 py-2.5 font-bold">#</th>
                        {displayCols.map(col => (
                          <th key={col} className={`px-3 py-2.5 font-bold ${highlightCols.has(col) ? 'text-primary' : ''}`}>
                            {col}
                          </th>
                        ))}
                        <th className="px-3 py-2.5 font-bold text-primary">燈號</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5 font-mono text-white/70">
                      {pagedRows.map((row, idx) => {
                        const globalIdx = page * PAGE_SIZE + idx;
                        const errPct = row.error_pct;
                        const rowBg = errPct !== null && errPct !== undefined
                          ? (errPct > 15 ? 'bg-red-500/[0.03]' : errPct > 5 ? 'bg-yellow-500/[0.02]' : '')
                          : '';
                        return (
                          <tr key={globalIdx} className={`hover:bg-white/[0.03] transition-colors ${rowBg}`}>
                            <td className="px-3 py-2 text-white/20">{globalIdx + 1}</td>
                            {displayCols.map(col => {
                              const val = row[col];
                              let cellClass = 'px-3 py-2';
                              if (col === 'predicted_EAC') cellClass += ' text-primary font-bold';
                              else if (col === 'EAC') cellClass += ' text-blue-400';
                              else if (col === 'error_pct') cellClass += ' hidden'; // shown in light column
                              return (
                                <td key={col} className={cellClass}>
                                  {val === null || val === undefined ? '—' : typeof val === 'number' ? Number(val).toFixed(4) : String(val)}
                                </td>
                              );
                            })}
                            <td className="px-3 py-2">
                              <ErrorLight pct={errPct} />
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-between mt-4 text-[10px] text-white/40">
                    <span>顯示 {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, rows.length)} / {rows.length}</span>
                    <div className="flex gap-1">
                      <button
                        onClick={() => setPage(p => Math.max(0, p - 1))}
                        disabled={page === 0}
                        className="px-3 py-1 rounded border border-white/10 hover:bg-white/5 disabled:opacity-20 transition-all"
                      >
                        上一頁
                      </button>
                      {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                        let pageNum;
                        if (totalPages <= 7) pageNum = i;
                        else if (page < 3) pageNum = i;
                        else if (page > totalPages - 4) pageNum = totalPages - 7 + i;
                        else pageNum = page - 3 + i;
                        return (
                          <button
                            key={pageNum}
                            onClick={() => setPage(pageNum)}
                            className={`px-2.5 py-1 rounded border transition-all ${page === pageNum ? 'border-primary text-primary bg-primary/10' : 'border-white/10 hover:bg-white/5'}`}
                          >
                            {pageNum + 1}
                          </button>
                        );
                      })}
                      <button
                        onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                        disabled={page >= totalPages - 1}
                        className="px-3 py-1 rounded border border-white/10 hover:bg-white/5 disabled:opacity-20 transition-all"
                      >
                        下一頁
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center space-y-4">
                <div className="size-20 rounded-full bg-white/5 mx-auto flex items-center justify-center">
                  <span className="material-symbols-outlined !text-4xl text-white/10">query_stats</span>
                </div>
                <p className="text-sm font-bold text-white/20 tracking-widest uppercase">上傳資料並選擇模型後即可開始預測</p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <div className="p-8 border-t border-white/10 bg-background-dark/95 flex justify-end gap-6 backdrop-blur-xl">
        <button onClick={onBack || onNavigateToDashboard} className="text-xs font-bold text-white/30 hover:text-white transition-colors">回模型訓練</button>
        <button onClick={onNavigateToDashboard} className="px-10 py-3 rounded-xl bg-white/5 border border-white/10 text-white font-bold text-xs hover:bg-white/10 hover:border-white/20 transition-all">返回首頁看板</button>
      </div>
    </div>
  );
}