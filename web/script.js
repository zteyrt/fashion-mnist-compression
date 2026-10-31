
/* ==========================================
   Fashion-MNIST AI 服饰识别系统
   完整 JavaScript 前端代码
========================================== */


// ==========================================
// 1. 获取 HTML 元素
// ==========================================

const imageInput = document.getElementById("imageInput");

const predictButton = document.getElementById("predictButton");
const compareButton = document.getElementById("compareButton");

const modelSelect = document.getElementById("modelSelect");

const result = document.getElementById("result");
const probabilityChart = document.getElementById("probabilityChart");
const comparisonResults = document.getElementById("comparisonResults");

const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");

const accuracyChart = document.getElementById("accuracyChart");
const parameterChart = document.getElementById("parameterChart");


// ==========================================
// 2. 全局变量
// ==========================================

let previewUrl = null;

const MAX_FILE_SIZE = 5 * 1024 * 1024;


// ==========================================
// 3. 图片合法性检查
// ==========================================

function validateImage(file) {

    if (!file) {
        return "请先选择一张图片";
    }

    if (!["image/png", "image/jpeg"].includes(file.type)) {
        return "请选择 PNG 或 JPG 图片";
    }

    if (file.size > MAX_FILE_SIZE) {
        return "图片大小不能超过 5MB";
    }

    return null;
}


// ==========================================
// 4. 图片选择与预览
// ==========================================

imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    // 清空旧的三模型对比结果
    comparisonResults.textContent =
        "请点击“同时对比三个模型”";

    // 释放上一张图片的预览地址
    if (previewUrl !== null) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    const errorMessage = validateImage(file);

    // 图片不存在或不符合要求
    if (errorMessage !== null) {

        previewContainer.hidden = true;
        imagePreview.removeAttribute("src");

        result.textContent = errorMessage;

        probabilityChart.textContent =
            "暂无预测概率数据";

        return;
    }

    // 创建新的图片预览地址
    previewUrl = URL.createObjectURL(file);

    imagePreview.src = previewUrl;
    previewContainer.hidden = false;

    // 清空旧的预测结果
    result.textContent =
        "图片已选择，请点击开始识别";

    probabilityChart.textContent =
        "请点击开始识别";
});


// ==========================================
// 5. 模型选择事件
// ==========================================

modelSelect.addEventListener("change", function () {

    result.textContent =
        "模型已切换，请点击开始识别";

    probabilityChart.textContent =
        "请点击开始识别";
});


// ==========================================
// 6. 绘制十类别概率柱状图
// ==========================================

function renderProbabilities(probabilities) {

    probabilityChart.replaceChildren();

    const sorted = Object.entries(probabilities).sort(
        (a, b) => b[1] - a[1]
    );

    sorted.forEach(function ([name, value], index) {

        const item = document.createElement("div");
        item.className = "probability-item";

        const label = document.createElement("div");
        label.className = "probability-label";

        const nameSpan = document.createElement("span");
        nameSpan.textContent = name;

        const valueSpan = document.createElement("span");
        valueSpan.textContent =
            Number(value).toFixed(2) + "%";

        label.append(nameSpan, valueSpan);

        const bar = document.createElement("div");
        bar.className = "probability-bar";

        const fill = document.createElement("div");
        fill.className = "probability-fill";

        // 最高概率使用绿色
        if (index === 0) {
            fill.classList.add("top");
        }

        fill.style.width =
            Math.max(0, Math.min(100, value)) + "%";

        bar.appendChild(fill);

        item.append(label, bar);

        probabilityChart.appendChild(item);
    });
}


// ==========================================
// 7. 单模型识别
// ==========================================

predictButton.addEventListener("click", async function () {

    const file = imageInput.files[0];

    const errorMessage = validateImage(file);

    if (errorMessage !== null) {

        result.textContent = errorMessage;

        probabilityChart.textContent =
            "暂无预测概率数据";

        return;
    }

    const formData = new FormData();

    formData.append("file", file);
    formData.append("model_name", modelSelect.value);

    result.textContent =
        "正在识别，请稍候...";

    probabilityChart.textContent =
        "正在计算预测概率...";

    predictButton.disabled = true;

    try {

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {

            const message =
                typeof data.detail === "string"
                    ? data.detail
                    : "识别失败";

            throw new Error(message);
        }

        result.textContent =
            "使用模型：" + data.model_display_name +
            " | 预测类别：" + data.predicted_class +
            " | 置信度：" +
            Number(data.confidence).toFixed(2) + "%";

        renderProbabilities(data.probabilities);

    } catch (error) {

        result.textContent =
            "发生错误：" + error.message;

        probabilityChart.textContent =
            "暂无预测概率数据";

        console.error("单模型识别失败：", error);

    } finally {

        predictButton.disabled = false;
    }
});


// ==========================================
// 8. 三模型同时对比
// ==========================================

compareButton.addEventListener("click", async function () {

    const file = imageInput.files[0];

    const errorMessage = validateImage(file);

    if (errorMessage !== null) {

        comparisonResults.textContent = errorMessage;
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    comparisonResults.textContent =
        "正在比较三个模型，请稍候...";

    compareButton.disabled = true;

    try {

        const response = await fetch("/compare", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {

            const message =
                typeof data.detail === "string"
                    ? data.detail
                    : "模型对比失败";

            throw new Error(message);
        }

        if (!Array.isArray(data.results)) {
            throw new Error("服务器返回的数据格式不正确");
        }

        renderComparison(data.results);

    } catch (error) {

        comparisonResults.textContent =
            "发生错误：" + error.message;

        console.error("三模型对比失败：", error);

    } finally {

        compareButton.disabled = false;
    }
});


// ==========================================
// 9. 绘制三模型对比卡片
// ==========================================

function renderComparison(results) {

    comparisonResults.replaceChildren();

    // 从实际返回结果中寻找原始模型的参数量
    const baselineModel = results.find(
        model => model.model_name === "baseline"
    );

    const baselineParameters = baselineModel
        ? Number(baselineModel.parameters)
        : null;

    results.forEach(function (model) {

        const card = document.createElement("div");
        card.className = "comparison-card";

        const title = document.createElement("h3");
        title.textContent = model.model_display_name;

        const prediction = document.createElement("p");
        prediction.textContent =
            "预测类别：" + model.predicted_class;

        const confidence = document.createElement("p");
        confidence.textContent =
            "置信度：" +
            Number(model.confidence).toFixed(2) + "%";

        const parameters = document.createElement("p");
        parameters.textContent =
            "参数量：" +
            Number(model.parameters).toLocaleString();

        const reduction = document.createElement("p");

        if (baselineParameters && baselineParameters > 0) {

            const reductionRate =
                (1 - Number(model.parameters) / baselineParameters) * 100;

            reduction.textContent =
                "参数减少：" + reductionRate.toFixed(2) + "%";

        } else {

            reduction.textContent =
                "参数减少：暂无数据";
        }

        card.append(
            title,
            prediction,
            confidence,
            parameters,
            reduction
        );

        comparisonResults.appendChild(card);
    });
}


// ==========================================
// 10. 模型压缩实验数据
// ==========================================

// 以下是此前在 Fashion-MNIST 测试集上得到的实验结果。
// 注意：accuracy 是整个测试集的准确率，
// 不是当前上传图片的预测置信度。

const compressionData = [
    {
        name: "原始 CNN",
        accuracy: 90.47,
        parameters: 206922
    },
    {
        name: "第一次剪枝",
        accuracy: 90.86,
        parameters: 205730
    },
    {
        name: "第二次剪枝",
        accuracy: 90.90,
        parameters: 154682
    }
];


// ==========================================
// 11. 通用指标柱状图函数
// ==========================================

function renderMetricChart(
    container,
    data,
    metric,
    maxValue,
    unit
) {

    container.replaceChildren();

    data.forEach(function (model) {

        const row = document.createElement("div");
        row.className = "metric-row";

        // 左侧名称与右侧指标数值
        const label = document.createElement("div");
        label.className = "metric-label";

        const nameSpan = document.createElement("span");
        nameSpan.textContent = model.name;

        const valueSpan = document.createElement("span");

        if (metric === "accuracy") {

            valueSpan.textContent =
                model.accuracy.toFixed(2) + unit;

        } else {

            valueSpan.textContent =
                model.parameters.toLocaleString() + unit;
        }

        label.append(nameSpan, valueSpan);

        // 创建柱状图
        const bar = document.createElement("div");
        bar.className = "metric-bar";

        const fill = document.createElement("div");
        fill.className = "metric-fill";

        // 根据数据计算柱子宽度
        const width = model[metric] / maxValue * 100;

        fill.style.width =
            Math.max(0, Math.min(100, width)) + "%";

        // 第二次剪枝模型高亮
        if (model.name === "第二次剪枝") {
            fill.classList.add("compressed");
        }

        bar.appendChild(fill);

        row.append(label, bar);

        container.appendChild(row);
    });
}


// ==========================================
// 12. 页面加载时绘制压缩分析图
// ==========================================

// 测试准确率对比
renderMetricChart(
    accuracyChart,
    compressionData,
    "accuracy",
    100,
    "%"
);

// 参数量对比
renderMetricChart(
    parameterChart,
    compressionData,
    "parameters",
    206922,
    ""
);
