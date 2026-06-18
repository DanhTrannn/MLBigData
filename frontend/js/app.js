const app = {
    currentStep: 1,
    currentLang: 'vi',
    i18n: {},

    async init() {
        await this.loadLanguage('vi');
    },

    async loadLanguage(lang) {
        try {
            const response = await fetch(`/i18n/${lang}.json`);
            this.i18n = await response.json();
            this.currentLang = lang;
            this.applyTranslations();
        } catch (e) {
            console.error('Failed to load language:', e);
        }
    },

    t(key) {
        const parts = key.split('.');
        let value = this.i18n;
        for (const part of parts) {
            if (value && typeof value === 'object') {
                value = value[part];
            } else {
                return key;
            }
        }
        return value || key;
    },

    applyTranslations() {
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            const text = this.t(key);
            if (text !== key) {
                if (el.tagName === 'INPUT' && el.type !== 'checkbox') {
                    el.placeholder = text;
                } else {
                    el.textContent = text;
                }
            }
        });

        document.getElementById('appTitle').textContent = this.t('app_title');
        document.getElementById('appSubtitle').textContent = this.t('app_subtitle');

        const langBtn = document.getElementById('langToggle');
        langBtn.textContent = this.currentLang === 'vi' ? 'English' : 'Tiếng Việt';

        if (resultRenderer.currentData) {
            resultRenderer.render(resultRenderer.currentData);
        }
    },

    toggleLanguage() {
        const newLang = this.currentLang === 'vi' ? 'en' : 'vi';
        this.loadLanguage(newLang);
    },

    nextStep() {
        if (this.currentStep === 1 && !form.validate()) return;

        const current = document.getElementById(`step${this.currentStep}`);
        if (this.currentStep < 3) {
            current.classList.remove('active');
            this.currentStep++;
            document.getElementById(`step${this.currentStep}`).classList.add('active');
            this.updateStepIndicators();
        }
    },

    prevStep() {
        if (this.currentStep > 1) {
            document.getElementById(`step${this.currentStep}`).classList.remove('active');
            this.currentStep--;
            document.getElementById(`step${this.currentStep}`).classList.add('active');
            this.updateStepIndicators();
        }
    },

    updateStepIndicators() {
        document.querySelectorAll('.step-dot').forEach((dot, i) => {
            dot.classList.remove('active', 'completed');
            if (i + 1 === this.currentStep) dot.classList.add('active');
            else if (i + 1 < this.currentStep) dot.classList.add('completed');
        });
    },

    async generate() {
        if (!form.validate()) {
            this.currentStep = 1;
            document.querySelectorAll('.wizard-step').forEach(s => s.classList.remove('active'));
            document.getElementById('step1').classList.add('active');
            this.updateStepIndicators();
            return;
        }

        const profile = form.getProfile();

        document.getElementById('profileWizard').classList.add('hidden');
        document.getElementById('loadingSection').classList.remove('hidden');

        try {
            const data = await api.recommendDay(profile);
            document.getElementById('loadingSection').classList.add('hidden');
            document.getElementById('resultPage').classList.remove('hidden');
            resultRenderer.render(data);
        } catch (error) {
            document.getElementById('loadingSection').classList.add('hidden');
            document.getElementById('errorSection').classList.remove('hidden');
            document.getElementById('errorTitle').textContent =
                error.message.includes('NO_SAFE') ? this.t('errors.no_candidate') :
                error.message.includes('NO_FEASIBLE') ? this.t('errors.no_plan') :
                'Error';
            document.getElementById('errorMessage').textContent = error.message;
        }
    },

    showWizard() {
        document.getElementById('resultPage').classList.add('hidden');
        document.getElementById('errorSection').classList.add('hidden');
        document.getElementById('loadingSection').classList.add('hidden');
        document.getElementById('profileWizard').classList.remove('hidden');
        this.currentStep = 1;
        document.querySelectorAll('.wizard-step').forEach(s => s.classList.remove('active'));
        document.getElementById('step1').classList.add('active');
        this.updateStepIndicators();
    },

    async sendFeedback(recId, foodId, eventType, mealType) {
        try {
            await api.submitFeedback({
                recommendation_id: recId,
                food_id: foodId,
                event_type: eventType,
                meal_type: mealType,
            });
            event.target.classList.add('active');
        } catch (e) {
            console.error('Feedback error:', e);
        }
    },

    async swapMeal(recId, foodId, mealType) {
        document.getElementById('resultPage').classList.add('hidden');
        document.getElementById('loadingSection').classList.remove('hidden');

        try {
            const data = await api.swapMeal(recId, foodId, mealType);
            document.getElementById('loadingSection').classList.add('hidden');
            document.getElementById('resultPage').classList.remove('hidden');
            resultRenderer.render(data);
        } catch (error) {
            document.getElementById('loadingSection').classList.add('hidden');
            document.getElementById('resultPage').classList.remove('hidden');
            alert(error.message);
        }
    },
};

document.addEventListener('DOMContentLoaded', () => app.init());
