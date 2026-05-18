// dashboard/static/script.js

async function fetchMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        
        document.getElementById('total-net-worth').innerText = `$${data.net_worth.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        
        const pnlEl = document.getElementById('unrealized-pnl');
        if (data.unrealized_pnl >= 0) {
            pnlEl.innerHTML = `<span class="material-symbols-outlined text-xs">arrow_upward</span> +${data.unrealized_pnl.toFixed(2)} U PnL`;
            pnlEl.className = "text-emerald-400 font-numeric-data text-numeric-data flex items-center justify-end gap-1";
        } else {
            pnlEl.innerHTML = `<span class="material-symbols-outlined text-xs">arrow_downward</span> ${data.unrealized_pnl.toFixed(2)} U PnL`;
            pnlEl.className = "text-rose-400 font-numeric-data text-numeric-data flex items-center justify-end gap-1";
        }

        document.getElementById('current-position').innerText = data.side;
        document.getElementById('btc-size').innerText = `${data.size} BTC`;
        document.getElementById('leverage').innerText = `x${data.leverage}`;

    } catch (e) {
        console.error("Failed to fetch metrics", e);
    }
}

async function fetchLogs() {
    try {
        const response = await fetch('/api/logs');
        const data = await response.json();
        const container = document.getElementById('live-logs-container');
        
        if (data.logs.length === 0) {
            container.innerHTML = '<div class="text-slate-500">Chưa có file log.</div>';
            return;
        }

        container.innerHTML = data.logs.map(log => {
            // Basic parsing of log line for styling (assuming standard format)
            // e.g. 2026-04-29 10:00:00 - INFO - Message
            let color = "text-emerald-500";
            if (log.includes("ERROR") || log.includes("WARN")) color = "text-amber-500";
            if (log.includes("SYNC")) color = "text-blue-400";
            
            return `
            <div class="flex gap-3">
                <span class="text-slate-300 break-words whitespace-pre-wrap">${log}</span>
            </div>`;
        }).join('');
    } catch (e) {
        console.error("Failed to fetch logs", e);
    }
}

async function fetchChart() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (data.times.length === 0) return;

        const trace = {
            x: data.times,
            y: data.net_worths,
            type: 'scatter',
            mode: 'lines',
            line: {color: '#58a6ff', width: 2, shape: 'spline'},
            fill: 'tozeroy',
            fillcolor: 'rgba(88, 166, 255, 0.1)'
        };

        const layout = {
            margin: {l: 40, r: 10, t: 10, b: 30},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: {showgrid: false, tickfont: {color: '#8b949e'}},
            yaxis: {showgrid: true, gridcolor: 'rgba(255,255,255,0.05)', tickfont: {color: '#8b949e'}},
            height: 220
        };

        Plotly.newPlot('chart-container', [trace], layout, {displayModeBar: false});
    } catch (e) {
        console.error("Failed to fetch chart", e);
    }
}

async function fetchInsight() {
    try {
        const response = await fetch('/api/insight');
        const data = await response.json();
        document.getElementById('strategy-insight').innerText = `"${data.insight}"`;
    } catch (e) {
        console.error("Failed to fetch insight", e);
    }
}

// Chat Functionality
const chatMessages = [];

function appendChatMessage(role, content) {
    const container = document.getElementById('chat-messages');
    
    // Xóa message placeholder nếu có
    if (container.innerHTML.includes("Hỏi AI về chiến lược")) {
        container.innerHTML = "";
    }

    const timeStr = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    let html = "";
    
    if (role === 'user') {
        html = `
        <div class="flex flex-col gap-1 items-end ml-auto max-w-[85%] mt-2">
            <div class="p-3 rounded-2xl bg-blue-600/20 border border-blue-500/30 text-blue-200 break-words">
                ${content}
            </div>
            <span class="text-[10px] text-slate-500 mr-2">${timeStr}</span>
        </div>`;
    } else {
        html = `
        <div class="flex flex-col gap-1 items-start max-w-[85%] mt-2">
            <div class="p-3 rounded-2xl bg-slate-800 text-slate-200 break-words">
                ${content}
            </div>
            <span class="text-[10px] text-slate-500 ml-2">${timeStr}</span>
        </div>`;
    }
    
    container.innerHTML += html;
    container.scrollTop = container.scrollHeight;
}

async function sendChat() {
    const inputEl = document.getElementById('chat-input');
    const msg = inputEl.value.trim();
    if (!msg) return;

    inputEl.value = "";
    appendChatMessage('user', msg);
    
    // Tạm thời disable input
    inputEl.disabled = true;
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: msg,
                history: chatMessages
            })
        });
        
        const data = await response.json();
        appendChatMessage('assistant', data.reply);
        
        // Lưu lịch sử
        chatMessages.push({role: 'user', content: msg});
        chatMessages.push({role: 'assistant', content: data.reply});
        
    } catch (e) {
        appendChatMessage('assistant', "Lỗi kết nối tới máy chủ AI.");
    } finally {
        inputEl.disabled = false;
        inputEl.focus();
    }
}

document.getElementById('chat-send').addEventListener('click', sendChat);
document.getElementById('chat-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendChat();
});

// Polling and initialization
setInterval(fetchMetrics, 3000);
setInterval(fetchLogs, 5000);
setInterval(fetchChart, 10000);
setInterval(fetchInsight, 60000); // 1 minute

// Initial load
fetchMetrics();
fetchLogs();
fetchChart();
fetchInsight();
