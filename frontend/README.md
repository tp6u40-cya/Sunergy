#  日光預 - 太陽光發電量預測平台

**日光預** 是一個專為綠能產業打造的 B2B 太陽能發電預測平台前端系統。透過現代化的暗黑模式介面 (Dark Mode UI)，提供案場管理、數據視覺化、以及完整的 AI 模型訓練流程體驗。

##  專案特色 (Features)

### 1. 沉浸式使用者體驗
* **動態登陸頁面**：具備呼吸燈效果的線稿 Logo 與流暢的進入動畫。
* **互動式使用者教學**：內建類似 PPT 的筆電情境導覽 (Scrollytelling)，引導使用者快速上手。
* **全站暗黑模式**：專為長時間監控設計的深色介面，搭配高對比的數據呈現。

### 2. 專業級儀表板 (Dashboard)
* **關鍵指標 (KPI)**：即時顯示發電效率 (kWh/kWp)、環境減碳效益 (ESG) 與預測準確度。
* **複合式圖表**：結合日照量背景與發電量曲線，並具備單位標示。
* **智慧排名**：根據昨日發電效率自動排序案場表現。

### 3. 完整的 AI 預測流程 (5-Step Wizard)
提供從資料上傳到報告輸出的完整引導流程：
1.  **資料上傳**：支援 CSV 拖曳上傳，自動辨識案場。
2.  **數據清理**：提供 9 種統計圖表 (直方圖/散佈圖) 檢視資料分佈，並具備一鍵移除離群值功能。
3.  **單位調整**：自動換算 kWh/m²、MJ/m² 等不同日照單位，並提供專業說明。
4.  **模型訓練**：
    * 支援 **LSTM, XGBoost, Random Forest, SVR** 四種模型。
    * **動態參數調整**：根據選擇的模型自動切換可調參數 (Epochs, Tree Depth, Kernel 等)。
    * **資料分割滑桿**：視覺化調整訓練集/測試集比例。
    * **訓練模擬**：具備真實感的 Loading 動畫與 Loss 數值跳動效果。
5.  **預測報告**：生成包含收益評估、減碳量與異常診斷的詳細報告。

### 4. 案場管理系統
* **手風琴式列表**：可展開查看詳細歷史預測紀錄。
* **批次操作**：支援多選刪除或單選快速跳轉預測。

## 技術堆疊 (Tech Stack)

* **Frontend Library**: [React](https://reactjs.org/) (Create React App)
* **Styling**: [Tailwind CSS](https://tailwindcss.com/) (v3)
* **Icons**: [Material Symbols](https://fonts.google.com/icons)
* **Charts**: Custom SVG Components (輕量化、無依賴的客製化圖表)
* **Animation**: CSS Keyframes & Tailwind Utilities

## 安裝與執行 (Getting Started)

請確保您的電腦已安裝 Node.js。

1.  **下載專案**
    ```bash
    git clone [https://github.com/您的帳號/daylight-pred-v2.git](https://github.com/您的帳號/daylight-pred-v2.git)
    cd daylight-pred-v2
    ```

2.  **安裝依賴套件**
    ```bash
    npm install
    ```

3.  **啟動開發伺服器**
    ```bash
    npm start
    ```
    瀏覽器將自動開啟 [http://localhost:3000](http://localhost:3000)。

## 案結構 (Project Structure)

```text
src/
├── components/
│   ├── Navbar.js          # 共用導覽列 (含頭像選單)
│   ├── LoginModal.js      # 登入/註冊懸浮視窗
│   └── CreateSiteModal.js # 新增案場懸浮視窗
├── pages/
│   ├── PublicHome.js      # 未登入首頁 (動畫 Logo)
│   ├── UserGuide.js       # 使用者教學 (PPT 風格導覽)
│   ├── Dashboard.js       # 主儀表板
│   ├── Sites.js           # 案場管理列表
│   ├── StartPredict.js    # 流程 1: 上傳資料
│   ├── DataCleaning.js    # 流程 2: 數據清理 (圖表)
│   ├── UnitAdjustment.js  # 流程 3: 單位調整
│   ├── ModelTraining.js   # 流程 4: 模型訓練 (參數設定)
│   └── PredictionReport.js# 流程 5: 預測報告
├── index.css              # Tailwind 引入與自定義動畫
└── App.js                 # 路由與全域狀態管理