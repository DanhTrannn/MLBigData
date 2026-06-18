const API_BASE = '/api/v1';

const api = {
    async recommendDay(profile) {
        const response = await fetch(`${API_BASE}/recommendations/day`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profile),
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            const detail = error.detail || error;
            throw new Error(detail.message || detail.error_code || `HTTP ${response.status}`);
        }
        return response.json();
    },

    async submitFeedback(feedback) {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(feedback),
        });
        return response.json();
    },

    async getDiseases() {
        const response = await fetch(`${API_BASE}/catalog/diseases`);
        return response.json();
    },

    async getAllergens() {
        const response = await fetch(`${API_BASE}/catalog/allergens`);
        return response.json();
    },

    async swapMeal(recommendationId, foodIdToRemove, mealType) {
        const response = await fetch(`${API_BASE}/recommendations/${recommendationId}/swap`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                recommendation_id: recommendationId,
                food_id_to_remove: foodIdToRemove,
                meal_type: mealType,
            }),
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail?.message || `HTTP ${response.status}`);
        }
        return response.json();
    },
};
