document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predict-form');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const simulateBtn = document.getElementById('simulate-btn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await predictProfile(new FormData(form));
    });

    simulateBtn.addEventListener('click', async () => {
        const formData = new FormData();
        formData.append('simulate', 'true');
        await predictProfile(formData);
    });

    async function predictProfile(formData) {
        loading.style.display = 'block';
        result.style.display = 'none';

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        loading.style.display = 'none';
        result.style.display = 'block';
        result.className = `result-box ${data.prediction.toLowerCase()}`;
        result.innerHTML = `
            <h2>Prediction: ${data.prediction}</h2>
            <p>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
            <p>Name: ${data.entry.features.name}</p>
            <p>Statuses: ${data.entry.features.statuses_count}</p>
            <p>Followers: ${data.entry.features.followers_count}</p>
            <p>Friends: ${data.entry.features.friends_count}</p>
            <p>Favourites: ${data.entry.features.favourites_count}</p>
            <p>Listed: ${data.entry.features.listed_count}</p>
            <p>Created At: ${data.entry.features.created_at}</p>
            <p>URL: ${data.entry.features.url || 'None'}</p>
            <p>Description: ${data.entry.features.description || 'None'}</p>
        `;

        if (window.location.pathname === '/history') {
            updateHistory();
        }
    }

    if (window.location.pathname === '/history') {
        updateHistory();
        setInterval(updateHistory, 2000);
    }
});

async function updateHistory() {
    const historyList = document.getElementById('history-list');
    const response = await fetch('/get_history');
    const history = await response.json();

    historyList.innerHTML = history.map(entry => `
        <div class="history-item ${entry.result.toLowerCase()}">
            <h3>${entry.features.name} (${entry.result})</h3>
            <p><strong>Confidence:</strong> ${(entry.confidence * 100).toFixed(2)}%</p>
            <p><strong>Statuses:</strong> ${entry.features.statuses_count}</p>
            <p><strong>Followers:</strong> ${entry.features.followers_count}</p>
            <p><strong>Friends:</strong> ${entry.features.friends_count}</p>
            <p><strong>Favourites:</strong> ${entry.features.favourites_count}</p>
            <p><strong>Listed:</strong> ${entry.features.listed_count}</p>
            <p><strong>Created At:</strong> ${entry.features.created_at}</p>
            <p><strong>URL:</strong> ${entry.features.url || 'None'}</p>
            <p><strong>Description:</strong> ${entry.features.description || 'None'}</p>
        </div>
    `).join('');
}
