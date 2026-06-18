const charts = {
    nutrientChart: null,

    renderNutrientChart(summary, targets) {
        const canvas = document.getElementById('nutrientChart');
        if (!canvas || typeof Chart === 'undefined') return;

        if (this.nutrientChart) {
            this.nutrientChart.destroy();
        }

        const labels = [
            app.t('result.calories'),
            app.t('result.protein'),
            app.t('result.carb'),
            app.t('result.fat'),
            app.t('result.sugar'),
        ];

        const actualValues = [
            summary.total_calories_kcal,
            summary.total_protein_g,
            summary.total_carb_g,
            summary.total_fat_g,
            summary.total_sugar_g,
        ];

        const targetValues = [
            targets.calories_kcal,
            targets.protein_g,
            targets.carb_g,
            targets.fat_g,
            targets.sugar_g,
        ];

        const maxValues = actualValues.map((a, i) => Math.max(a, targetValues[i]) * 1.2);
        const actualPct = actualValues.map((v, i) => maxValues[i] > 0 ? (v / maxValues[i]) * 100 : 0);
        const targetPct = targetValues.map((v, i) => maxValues[i] > 0 ? (v / maxValues[i]) * 100 : 0);

        this.nutrientChart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: app.t('result.actual'),
                        data: actualPct,
                        backgroundColor: 'rgba(21, 101, 192, 0.7)',
                        borderRadius: 6,
                    },
                    {
                        label: app.t('result.target'),
                        data: targetPct,
                        backgroundColor: 'rgba(21, 101, 192, 0.2)',
                        borderColor: 'rgba(21, 101, 192, 0.5)',
                        borderWidth: 1,
                        borderRadius: 6,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const idx = context.dataIndex;
                                const ds = context.datasetIndex;
                                if (ds === 0) return `${app.t('result.actual')}: ${actualValues[idx]}`;
                                return `${app.t('result.target')}: ${targetValues[idx]}`;
                            },
                        },
                    },
                },
                scales: {
                    y: { display: false },
                    x: { grid: { display: false } },
                },
            },
        });
    },
};
