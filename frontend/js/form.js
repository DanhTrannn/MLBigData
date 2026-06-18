const form = {
    getProfile() {
        const diseases = Array.from(document.querySelectorAll('input[name="diseases"]:checked'))
            .map(cb => cb.value);
        const allergies = Array.from(document.querySelectorAll('input[name="allergies"]:checked'))
            .map(cb => cb.value);
        const dislikedRaw = document.getElementById('disliked_ingredients').value.trim();
        const disliked_ingredients = dislikedRaw
            ? dislikedRaw.split(',').map(s => s.trim().toLowerCase()).filter(Boolean)
            : [];
        const selectedTags = Array.from(document.querySelectorAll('.chip.selected'))
            .map(chip => chip.dataset.tag);
        const budgetRaw = document.getElementById('budget_per_day').value;
        const budget_per_day = budgetRaw ? parseFloat(budgetRaw) : null;
        const sex = document.getElementById('sex').value;
        const height = parseFloat(document.getElementById('height_cm').value);
        const weight = parseFloat(document.getElementById('weight_kg').value);

        return {
            age: parseInt(document.getElementById('age').value),
            height_cm: height,
            weight_kg: weight,
            sex: sex,
            activity_level: document.getElementById('activity_level').value,
            goal: document.getElementById('goal').value,
            diseases: diseases,
            allergies: allergies,
            disliked_ingredients: disliked_ingredients,
            preferred_tags: selectedTags,
            budget_per_day: budget_per_day,
            selected_likes: [],
            lang: app.currentLang,
        };
    },

    validate() {
        let valid = true;
        const age = document.getElementById('age');
        const height = document.getElementById('height_cm');
        const weight = document.getElementById('weight_kg');

        document.querySelectorAll('.error-msg').forEach(el => el.textContent = '');

        if (!age.value || parseInt(age.value) < 1 || parseInt(age.value) > 120) {
            document.getElementById('age-error').textContent = app.t('errors.age_range');
            valid = false;
        }
        if (!height.value || parseFloat(height.value) < 50 || parseFloat(height.value) > 250) {
            document.getElementById('height_cm-error').textContent = app.t('errors.height_range');
            valid = false;
        }
        if (!weight.value || parseFloat(weight.value) < 10 || parseFloat(weight.value) > 300) {
            document.getElementById('weight_kg-error').textContent = app.t('errors.weight_range');
            valid = false;
        }
        return valid;
    },

    updateBMI() {
        const h = parseFloat(document.getElementById('height_cm').value);
        const w = parseFloat(document.getElementById('weight_kg').value);
        if (h > 0 && w > 0) {
            const bmi = (w / ((h / 100) ** 2)).toFixed(1);
            document.getElementById('bmiValue').textContent = bmi;
        } else {
            document.getElementById('bmiValue').textContent = '--';
        }
    },

    initTagChips() {
        document.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                chip.classList.toggle('selected');
            });
        });
    },
};

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('height_cm').addEventListener('input', form.updateBMI);
    document.getElementById('weight_kg').addEventListener('input', form.updateBMI);
    form.initTagChips();
});
