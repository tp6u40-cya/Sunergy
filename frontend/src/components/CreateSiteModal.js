// src/components/CreateSiteModal.js
import React, { useState } from 'react';

export default function CreateSiteModal({ onClose, onSubmit }) {
  // 防止點擊內部時關閉
  const handleModalClick = (e) => {
    e.stopPropagation();
  };

  // ➜ 正確的資料表欄位
  const [formData, setFormData] = useState({
    site_code: "",
    site_name: "",
    location: ""
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.id]: e.target.value });
  };

  const handleSubmit = () => {
    // 基本驗證（避免空白）
    if (!formData.site_code || !formData.site_name || !formData.location) {
      alert("請完整填寫案場代號、案場名稱與地點");
      return;
    }

    onSubmit(formData);  // 將資料回傳給 App.js → 呼叫 API
  };

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 backdrop-blur-sm transition-opacity"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-md transform rounded-xl border border-white/10 bg-[#1E1E1E] p-6 shadow-2xl transition-all"
        onClick={handleModalClick}
      >

        {/* 標題 */}
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-xl font-bold text-white">建立新的案場</h2>
          <button
            onClick={onClose}
            className="text-white/50 hover:text-white transition-colors"
          >
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>

        {/* ------------------ 表單 ------------------ */}
        <div className="flex flex-col gap-4">

          {/* 案場代號 */}
          <div className="flex flex-col gap-2">
            <label htmlFor="site_code" className="text-sm font-medium text-white/80">
              案場代號
            </label>
            <input
              type="text"
              id="site_code"
              placeholder="請輸入案場代號，例如：A001"
              value={formData.site_code}
              onChange={handleChange}
              className="w-full rounded-lg border border-white/10 bg-black/20 px-4 py-3 text-sm text-white placeholder-white/30 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary transition-all"
            />
          </div>

          {/* 案場名稱 */}
          <div className="flex flex-col gap-2">
            <label htmlFor="site_name" className="text-sm font-medium text-white/80">
              案場名稱
            </label>
            <input
              type="text"
              id="site_name"
              placeholder="請輸入案場名稱"
              value={formData.site_name}
              onChange={handleChange}
              className="w-full rounded-lg border border-white/10 bg-black/20 px-4 py-3 text-sm text-white placeholder-white/30 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary transition-all"
            />
          </div>

          {/* 地點 */}
          <div className="flex flex-col gap-2">
            <label htmlFor="location" className="text-sm font-medium text-white/80">
              地點
            </label>
            <input
              type="text"
              id="location"
              placeholder="例如：嘉義縣太保市"
              value={formData.location}
              onChange={handleChange}
              className="w-full rounded-lg border border-white/10 bg-black/20 px-4 py-3 text-sm text-white placeholder-white/30 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary transition-all"
            />
          </div>

        </div>

        {/* 按鈕 */}
        <div className="mt-8 flex justify-end">
          <button
            onClick={handleSubmit}
            className="rounded-lg bg-primary px-6 py-2.5 text-sm font-bold text-background-dark hover:scale-105 transition-transform"
          >
            確認新增
          </button>
        </div>

      </div>
    </div>
  );
}
