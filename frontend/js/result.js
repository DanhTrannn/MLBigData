const resultRenderer = {
    currentData: null,

    render(data) {
        this.currentData = data;
        document.getElementById('ruleVersion').textContent =
            `${app.t('result.rule_version')}: ${data.rule_version}`;
        document.getElementById('modelVersion').textContent =
            `${app.t('result.model_version')}: ${data.model_version}`;

        this.renderSummary(data.summary, data.targets);
        this.renderMeals(data.meals, data.recommendation_id);
        if (typeof charts !== 'undefined') {
            charts.renderNutrientChart(data.summary, data.targets);
        }
    },

    renderSummary(summary, targets) {
        const grid = document.getElementById('summaryGrid');
        const items = [
            { label: app.t('result.calories'), value: `${summary.total_calories_kcal}`, unit: 'kcal', target: `${targets.calories_kcal}` },
            { label: app.t('result.protein'), value: `${summary.total_protein_g}`, unit: 'g', target: `${targets.protein_g}` },
            { label: app.t('result.carb'), value: `${summary.total_carb_g}`, unit: 'g', target: `${targets.carb_g}` },
            { label: app.t('result.fat'), value: `${summary.total_fat_g}`, unit: 'g', target: `${targets.fat_g}` },
            { label: app.t('result.sugar'), value: `${summary.total_sugar_g}`, unit: 'g', target: `${targets.sugar_g}` },
            { label: app.t('result.fiber'), value: `${summary.total_fiber_g}`, unit: 'g' },
            { label: app.t('result.sodium'), value: `${summary.total_sodium_mg}`, unit: 'mg', target: `${targets.sodium_mg}` },
            { label: app.t('result.cost'), value: `${Math.round(summary.estimated_cost).toLocaleString()}`, unit: '₫' },
        ];

        grid.innerHTML = items.map(item => `
            <div class="summary-item">
                <div class="label">${item.label}</div>
                <div class="value">${item.value}<small> ${item.unit}</small></div>
                ${item.target ? `<div class="target">${app.t('result.target')}: ${item.target} ${item.unit}</div>` : ''}
            </div>
        `).join('');

        const statusEl = document.getElementById('constraintStatus');
        if (summary.constraint_status === 'pass') {
            statusEl.className = 'constraint-status pass';
            statusEl.textContent = app.t('result.constraint_pass');
        } else {
            statusEl.className = 'constraint-status warn';
            statusEl.textContent = app.t('result.constraint_warn');
        }
    },

    renderMeals(meals, recId) {
        const grid = document.getElementById('mealsGrid');
        const mealLabels = {
            breakfast: app.t('result.breakfast'),
            lunch: app.t('result.lunch'),
            dinner: app.t('result.dinner'),
        };

        grid.innerHTML = meals.map(meal => `
            <div class="meal-card" data-food-id="${meal.food_id}" data-meal-type="${meal.meal_type}">
                <span class="meal-type">${mealLabels[meal.meal_type] || meal.meal_type}</span>
                <div class="food-name">${meal.food_name}</div>
                <table class="nutrition-table">
                    <tr><td>${app.t('result.calories')}</td><td>${meal.nutrition.calories_kcal} kcal</td></tr>
                    <tr><td>${app.t('result.protein')}</td><td>${meal.nutrition.protein_g}g</td></tr>
                    <tr><td>${app.t('result.carb')}</td><td>${meal.nutrition.carb_g}g</td></tr>
                    <tr><td>${app.t('result.fat')}</td><td>${meal.nutrition.fat_g}g</td></tr>
                    <tr><td>${app.t('result.sodium')}</td><td>${meal.nutrition.sodium_mg}mg</td></tr>
                </table>
                <div class="cost">${app.t('result.cost')}: ${Math.round(meal.cost).toLocaleString()}₫</div>
                <div class="score">Score: ${(meal.suitability_score * 100).toFixed(0)}%</div>
                <div class="explanation-panel">
                    <h4>${app.t('result.explanations')}</h4>
                    <ul>
                        ${meal.explanations.map(e => `<li>${e}</li>`).join('')}
                    </ul>
                </div>
                <div class="feedback-buttons">
                    <button class="btn-feedback" onclick="app.sendFeedback('${recId}','${meal.food_id}','like','${meal.meal_type}')">${app.t('result.like')}</button>
                    <button class="btn-feedback" onclick="app.sendFeedback('${recId}','${meal.food_id}','dislike','${meal.meal_type}')">${app.t('result.dislike')}</button>
                    <button class="btn-feedback" onclick="app.sendFeedback('${recId}','${meal.food_id}','eaten','${meal.meal_type}')">${app.t('result.eaten')}</button>
                </div>
                <button class="swap-btn" onclick="app.swapMeal('${recId}','${meal.food_id}','${meal.meal_type}')">
                    ${app.t('buttons.swap')}
                </button>
            </div>
        `).join('');
    },
};
