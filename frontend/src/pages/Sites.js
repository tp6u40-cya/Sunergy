// src/pages/Sites.js
import React, { useState, useEffect } from "react";
import Navbar from "../components/Navbar";

export default function Sites({
  onNavigateToDashboard,
  onNavigateToPredict,
  onNavigateToSites,
  onLogout,
  onOpenCreateSite,
  user,
}) {
  const [sites, setSites] = useState([]);
  const [expandedSiteId, setExpandedSiteId] = useState(null);
  const [selectedSiteIds, setSelectedSiteIds] = useState([]);

  // --------------------------------------
  // 🟦 進入頁面時：向後端讀取案場列表
  // --------------------------------------
  useEffect(() => {
    if (!user) return;

    const fetchSites = async () => {
      try {
        const res = await fetch(
          `http://127.0.0.1:8000/site/list?user_id=${user.user_id}`
        );

        const data = await res.json();
        console.log("讀到案場列表:", data);

        setSites(data);
      } catch (err) {
        console.error("讀取案場列表失敗:", err);
      }
    };

    fetchSites();
  }, [user]);

  // 當未從 props 取得 user 時，嘗試從 localStorage 讀取並載入站點清單
  useEffect(() => {
    if (user) return; // 交由上方 effect 處理
    const stored = JSON.parse(localStorage.getItem("user") || "{}");
    const uid = stored?.user_id;
    if (!uid) return;

    const fetchSites = async () => {
      try {
        const res = await fetch(
          `http://127.0.0.1:8000/site/list?user_id=${uid}`
        );
        const data = await res.json();
        setSites(Array.isArray(data) ? data : []);
      } catch (err) {
        console.error("load sites from localStorage user failed", err);
      }
    };

    fetchSites();
  }, [user]);

  // 展開 / 收合卡片
  const toggleExpand = (id) => {
    setExpandedSiteId(expandedSiteId === id ? null : id);
  };

  // 勾選案場
  const toggleSelect = (id) => {
    if (selectedSiteIds.includes(id)) {
      setSelectedSiteIds(selectedSiteIds.filter((x) => x !== id));
    } else {
      setSelectedSiteIds([...selectedSiteIds, id]);
    }
  };

  // 批次刪除（前端）
  const handleBatchDelete = () => {
    const confirmDelete = window.confirm(
      `確定刪除 ${selectedSiteIds.length} 個案場？`
    );
    if (!confirmDelete) return;

    setSites(sites.filter((s) => !selectedSiteIds.includes(s.site_id)));
    setSelectedSiteIds([]);
  };

  return (
    <div className="min-h-screen w-full bg-background-dark text-white flex flex-col">
      <Navbar
        activePage="site"
        onNavigateToDashboard={onNavigateToDashboard}
        onNavigateToPredict={onNavigateToPredict}
        onNavigateToSites={onNavigateToSites}
        onLogout={onLogout}
      />

      {/* 主要內容 */}
      <main className="flex-1 w-full max-w-7xl mx-auto p-6 py-10">
        {/* 標題 + 新增按鈕 */}
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold text-white">我的案場管理</h1>

          <button
            onClick={onOpenCreateSite}
            className="flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-bold text-background-dark transition-transform hover:scale-105"
          >
            <span className="material-symbols-outlined !text-xl">add</span>
            新增案場
          </button>
        </div>

        {/* 案場列表 */}
        <div className="flex flex-col gap-4">
          {sites.length === 0 && (
            <p className="text-white/60 text-sm">
              目前沒有案場，請建立一個新案場。
            </p>
          )}

          {sites.map((site) => (
            <div
              key={site.site_id}
              className={`rounded-xl border transition-all duration-300 ${
                expandedSiteId === site.site_id
                  ? "border-primary/50 bg-white/[.02]"
                  : "border-white/10 bg-white/[.01] hover:bg-white/[.03]"
              }`}
            >
              {/* 卡片標頭 */}
              <div className="flex items-center justify-between p-4 sm:p-6">
                <div className="flex items-center gap-6">
                  {/* 勾選 */}
                  <div className="relative flex items-center">
                    <input
                      type="checkbox"
                      checked={selectedSiteIds.includes(site.site_id)}
                      onChange={() => toggleSelect(site.site_id)}
                      className="peer size-5 cursor-pointer appearance-none rounded border border-white/30 bg-transparent checked:border-primary checked:bg-primary transition-all"
                    />
                    <span className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-background-dark opacity-0 peer-checked:opacity-100 material-symbols-outlined !text-base font-bold">
                      check
                    </span>
                  </div>

                  {/* Icon + 資訊 */}
                  <div className="flex items-center gap-4">
                    <div className="flex size-12 items-center justify-center rounded-lg bg-primary/10 text-primary">
                      <span className="material-symbols-outlined !text-3xl">
                        solar_power
                      </span>
                    </div>

                    <div>
                      <h3 className="text-lg font-bold">{site.site_name}</h3>
                      <p className="text-sm text-white/50">{site.location}</p>
                    </div>
                  </div>
                </div>

                {/* 動作按鈕 */}
                <div className="flex items-center gap-2">
                  <button className="flex size-10 items-center justify-center rounded-full text-white/50 hover:bg-white/10 hover:text-white transition-colors">
                    <span className="material-symbols-outlined !text-xl">
                      edit
                    </span>
                  </button>

                  <button
                    onClick={() => toggleExpand(site.site_id)}
                    className={`flex size-10 items-center justify-center rounded-full text-white/50 hover:bg-white/10 hover:text-white transition-transform duration-300 ${
                      expandedSiteId === site.site_id ? "rotate-180" : ""
                    }`}
                  >
                    <span className="material-symbols-outlined !text-2xl">
                      expand_more
                    </span>
                  </button>
                </div>
              </div>

              {/* 展開內容 */}
              {expandedSiteId === site.site_id && (
                <div className="border-t border-white/10 bg-white/[.02] p-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    <div>
                      <p className="text-xs text-white/40 mb-1">案場代號</p>
                      <p className="text-sm font-mono">{site.site_code}</p>
                    </div>

                    <div>
                      <p className="text-xs text-white/40 mb-1">建立日期</p>
                      <p className="text-sm font-mono">
                        {site.created_at?.slice(0, 10)}
                      </p>
                    </div>

                    <div>
                      <p className="text-xs text-white/40 mb-1">使用者 ID</p>
                      <p className="text-sm font-mono">{site.user_id}</p>
                    </div>
                  </div>

                  <div className="mt-6 flex justify-end gap-3">
                    <button
                      onClick={async () => {
                        try {
                          const dataId = localStorage.getItem("lastDataId");
                          if (!dataId) {
                            alert("目前沒有可下載的資料");
                            return;
                          }
                          const res = await fetch(`http://127.0.0.1:8000/site/download?data_id=${dataId}`);
                          const data = await res.json();
                          if (!res.ok) throw new Error(data?.detail || "下載失敗");
                          alert(`檔名: ${data.file_name}\n大小: ${data.size} bytes`);
                        } catch (e) {
                          alert("暫無可下載的原始檔或下載失敗");
                        }
                      }}
                      className="px-4 py-2 rounded border border-white/10 text-xs text-white/70 hover:text-white hover:border-white/40 transition-colors"
                    >
                      下載資料
                    </button>
                    <button className="px-4 py-2 rounded border border-white/10 text-xs text-white/60 hover:text-red-400 hover:border-red-400/50 transition-colors">
                      刪除案場
                    </button>

                    <button
                      onClick={onNavigateToPredict}
                      className="px-4 py-2 rounded bg-white/10 text-xs font-bold hover:bg-white/20 transition-colors flex items-center gap-2"
                    >
                      以此案場開始預測
                      <span className="material-symbols-outlined !text-sm">
                        arrow_forward
                      </span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </main>

      {/* Sticky Footer：批次刪除 */}
      {selectedSiteIds.length > 0 && (
        <div className="sticky bottom-0 w-full border-t border-white/10 bg-[#1E1E1E] p-4 px-6 z-40 shadow-2xl">
          <div className="mx-auto flex max-w-7xl items-center justify-between">
            <div className="text-sm">
              已選取{" "}
              <span className="text-primary font-bold">
                {selectedSiteIds.length}
              </span>{" "}
              個案場
            </div>

            <button
              onClick={handleBatchDelete}
              className="flex items-center justify-center gap-2 rounded-lg border border-red-500/50 text-red-400 px-6 py-2 text-sm font-bold hover:bg-red-500/10 transition-colors"
            >
              <span className="material-symbols-outlined !text-lg">delete</span>
              批次刪除
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
