const fileInput = document.getElementById("question-file");
const questionContainer = document.getElementById("question-container");
const questionList = document.getElementById("question-list");
const totalQuestions = document.getElementById("total-questions");
const countdownEl = document.getElementById("countdown");

const QUESTION_TYPE_MAP = {
  单选: "单选题",
  多选: "多选题",
  判断: "判断题",
  单选题: "单选题",
  多选题: "多选题",
  判断题: "判断题",
};

let questions = [];
let activeQuestionId = null;
let activeObserver = null;

function setCountdown(durationMinutes = 120) {
  const endTime = Date.now() + durationMinutes * 60 * 1000;
  function tick() {
    const diff = endTime - Date.now();
    if (diff <= 0) {
      countdownEl.textContent = "00:00:00";
      return;
    }
    const hours = Math.floor(diff / 1000 / 60 / 60)
      .toString()
      .padStart(2, "0");
    const minutes = Math.floor((diff / 1000 / 60) % 60)
      .toString()
      .padStart(2, "0");
    const seconds = Math.floor((diff / 1000) % 60)
      .toString()
      .padStart(2, "0");
    countdownEl.textContent = `${hours}:${minutes}:${seconds}`;
    requestAnimationFrame(tick);
  }
  tick();
}

function createEmptyState() {
  questionContainer.innerHTML = `
    <div class="question-card empty-state">
      暂无题目，请在左侧导入题库 CSV 文件。
    </div>
  `;
  questionList.innerHTML = "";
  totalQuestions.textContent = "0";
}

function buildQuestionNav() {
  questionList.innerHTML = "";
  questions.forEach((q, index) => {
    const pill = document.createElement("button");
    pill.className = `question-pill ${q.id === activeQuestionId ? "active" : ""}`;
    pill.textContent = index + 1;
    pill.addEventListener("click", () => {
      scrollToQuestion(q.id);
    });
    questionList.appendChild(pill);
  });
}

function scrollToQuestion(id) {
  const card = document.querySelector(`[data-question-id="${id}"]`);
  if (card) {
    card.scrollIntoView({ behavior: "smooth", block: "start" });
    setActiveQuestion(id);
  }
}

function setActiveQuestion(id) {
  activeQuestionId = id;
  document.querySelectorAll(".question-pill").forEach((pill, index) => {
    pill.classList.toggle("active", questions[index].id === id);
  });
}

function createOption(optionKey, content, name, type) {
  if (!content) return null;
  const option = document.createElement("label");
  option.className = "option";

  const input = document.createElement("input");
  input.type = type === "多选题" ? "checkbox" : "radio";
  input.name = name;
  input.value = optionKey;

  const badge = document.createElement("span");
  badge.className = "option-label";
  badge.textContent = optionKey;

  const text = document.createElement("span");
  text.className = "option-content";
  text.textContent = content;

  option.appendChild(input);
  option.appendChild(badge);
  option.appendChild(text);

  option.addEventListener("click", () => {
    setActiveQuestion(name);
  });

  return option;
}

function renderQuestions() {
  if (!questions.length) {
    createEmptyState();
    return;
  }
  questionContainer.innerHTML = "";

  questions.forEach((question, index) => {
    const card = document.createElement("article");
    card.className = "question-card";
    card.dataset.questionId = question.id;

    const header = document.createElement("div");
    header.className = "question-header";

    const title = document.createElement("h3");
    title.textContent = `${index + 1}. ${question.text}`;

    const type = document.createElement("span");
    type.className = "question-type";
    type.textContent = QUESTION_TYPE_MAP[question.type] || question.type;

    header.appendChild(title);
    header.appendChild(type);

    const optionsList = document.createElement("div");
    optionsList.className = "option-list";

    const optionKeys = [
      "选项A",
      "选项B",
      "选项C",
      "选项D",
      "选项E",
      "选项F",
      "选项G",
      "选项H",
    ];

    optionKeys.forEach((key, optionIndex) => {
      const option = createOption(
        String.fromCharCode(65 + optionIndex),
        question.options[key],
        question.id,
        QUESTION_TYPE_MAP[question.type] || question.type
      );
      if (option) {
        optionsList.appendChild(option);
      }
    });

    if (QUESTION_TYPE_MAP[question.type] === "判断题" && optionsList.children.length === 0) {
      const trueOption = createOption("正确", "正确", question.id, "判断题");
      const falseOption = createOption("错误", "错误", question.id, "判断题");
      optionsList.appendChild(trueOption);
      optionsList.appendChild(falseOption);
    }

    card.appendChild(header);
    card.appendChild(optionsList);
    questionContainer.appendChild(card);
  });

  totalQuestions.textContent = questions.length;
  buildQuestionNav();
  observeQuestions();
}

function observeQuestions() {
  if (activeObserver) {
    activeObserver.disconnect();
  }
  activeObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
      if (visible.length) {
        const id = visible[0].target.dataset.questionId;
        setActiveQuestion(id);
      }
    },
    {
      root: null,
      threshold: [0.3, 0.6, 0.9],
    }
  );

  document.querySelectorAll(".question-card").forEach((card) => {
    activeObserver.observe(card);
  });
}

function normaliseType(rawType = "") {
  const type = rawType.trim();
  if (/^单选/.test(type)) return "单选题";
  if (/^多选/.test(type)) return "多选题";
  if (/^判断/.test(type)) return "判断题";
  return type || "单选题";
}

function transformRow(row, index) {
  const text = (row["题干"] || "").trim();
  const type = normaliseType(row["题型"] || "");
  if (!text || !type) {
    console.warn(`第 ${index + 1} 行缺少题干或题型，已跳过。`);
    return null;
  }
  const options = {};
  Object.keys(row).forEach((key) => {
    if (key.startsWith("选项") && row[key]) {
      options[key] = row[key].trim();
    }
  });

  return {
    id: `q${index + 1}`,
    text,
    type,
    options,
    answer: (row["正确答案"] || "").trim(),
  };
}

function handleFile(file) {
  if (!file) return;
  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    encoding: "UTF-8",
    complete: (results) => {
      const transformed = results.data
        .map((row, index) => transformRow(row, index))
        .filter(Boolean);
      if (!transformed.length) {
        createEmptyState();
        return;
      }
      questions = transformed;
      activeQuestionId = questions[0].id;
      renderQuestions();
      setActiveQuestion(activeQuestionId);
    },
    error: (error) => {
      console.error(error);
      createEmptyState();
    },
  });
}

fileInput?.addEventListener("change", (event) => {
  const [file] = event.target.files;
  handleFile(file);
});

createEmptyState();
setCountdown();
