const TelegramBot = require("node-telegram-bot-api");
const Anthropic = require("@anthropic-ai/sdk");
const { createClient } = require("@supabase/supabase-js");
const https = require("https");
require("dotenv").config();

// ── Clients ───────────────────────────────────────────────────────────────────
const bot = new TelegramBot(process.env.TELEGRAM_BOT_TOKEN, { polling: true });
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

// ── Helpers ───────────────────────────────────────────────────────────────────
function today() {
  return new Date().toISOString().split("T")[0];
}

function formatMeal(meal) {
  return (
    `🍽 *${meal.food_name}*\n` +
    `• Calories: ${meal.calories} kcal\n` +
    `• Protein: ${meal.protein}g\n` +
    `• Carbs: ${meal.carbs}g\n` +
    `• Fat: ${meal.fat}g\n` +
    `• Fiber: ${meal.fiber}g\n` +
    `• Sugar: ${meal.sugar}g`
  );
}

function formatTotals(rows) {
  return rows.reduce(
    (acc, m) => {
      acc.calories += m.calories || 0;
      acc.protein += m.protein || 0;
      acc.carbs += m.carbs || 0;
      acc.fat += m.fat || 0;
      acc.fiber += m.fiber || 0;
      acc.sugar += m.sugar || 0;
      return acc;
    },
    { calories: 0, protein: 0, carbs: 0, fat: 0, fiber: 0, sugar: 0 }
  );
}

// Download image from URL and return as base64
function fetchImageAsBase64(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      const chunks = [];
      res.on("data", (chunk) => chunks.push(chunk));
      res.on("end", () => {
        const buffer = Buffer.concat(chunks);
        resolve(buffer.toString("base64"));
      });
      res.on("error", reject);
    });
  });
}

const NUTRITION_PROMPT = `You are a nutrition expert. Respond ONLY with a valid JSON object, no markdown, no explanation:
{
  "food_name": "descriptive name of the food",
  "calories": 0,
  "protein": 0,
  "carbs": 0,
  "fat": 0,
  "fiber": 0,
  "sugar": 0,
  "notes": "brief note about portion size assumptions"
}
All macros in grams. Calories in kcal. Be realistic with portion estimates.`;

// ── Analyze food image with Claude Vision ─────────────────────────────────────
async function analyzeFoodImage(imageUrl) {
  const base64Image = await fetchImageAsBase64(imageUrl);

  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 500,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/jpeg",
              data: base64Image,
            },
          },
          {
            type: "text",
            text: `Analyze this food image and estimate its nutritional content.\n\n${NUTRITION_PROMPT}`,
          },
        ],
      },
    ],
  });

  const text = response.content[0].text.trim();
  return JSON.parse(text);
}

// ── Analyze food from text description ───────────────────────────────────────
async function analyzeFoodText(description) {
  const response = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 500,
    messages: [
      {
        role: "user",
        content: `Estimate the nutritional content for: "${description}"\n\n${NUTRITION_PROMPT}`,
      },
    ],
  });

  const text = response.content[0].text.trim();
  return JSON.parse(text);
}

// ── AI summary of food diary ──────────────────────────────────────────────────
async function getAISummary(meals, period) {
  const mealList = meals
    .map(
      (m) =>
        `${m.date} - ${m.food_name}: ${m.calories}cal, ${m.protein}g protein, ${m.carbs}g carbs, ${m.fat}g fat`
    )
    .join("\n");

  const totals = formatTotals(meals);

  const response = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 400,
    messages: [
      {
        role: "user",
        content: `Here is my food diary for ${period}:\n\n${mealList}\n\nTotals: ${totals.calories} calories, ${totals.protein}g protein, ${totals.carbs}g carbs, ${totals.fat}g fat, ${totals.fiber}g fiber, ${totals.sugar}g sugar\n\nGive me a brief, friendly nutritional analysis. Mention what I'm doing well and one thing to improve. Keep it under 100 words.`,
      },
    ],
  });

  return response.content[0].text;
}

// ── Supabase helpers ──────────────────────────────────────────────────────────
async function saveMeal(chatId, nutrition) {
  const { data, error } = await supabase
    .from("meals")
    .insert([{
      chat_id: chatId.toString(),
      food_name: nutrition.food_name,
      calories: nutrition.calories,
      protein: nutrition.protein,
      carbs: nutrition.carbs,
      fat: nutrition.fat,
      fiber: nutrition.fiber,
      sugar: nutrition.sugar,
      notes: nutrition.notes,
      logged_at: new Date().toISOString(),
      date: today(),
    }])
    .select()
    .single();

  if (error) throw new Error(error.message);
  return data;
}

async function getTodaysMeals(chatId) {
  const { data, error } = await supabase
    .from("meals")
    .select("*")
    .eq("chat_id", chatId.toString())
    .eq("date", today())
    .order("logged_at", { ascending: true });

  if (error) throw new Error(error.message);
  return data;
}

async function getMealsForRange(chatId, startDate, endDate) {
  const { data, error } = await supabase
    .from("meals")
    .select("*")
    .eq("chat_id", chatId.toString())
    .gte("date", startDate)
    .lte("date", endDate)
    .order("logged_at", { ascending: true });

  if (error) throw new Error(error.message);
  return data;
}

async function deleteLastMeal(chatId) {
  const { data } = await supabase
    .from("meals")
    .select("id")
    .eq("chat_id", chatId.toString())
    .order("logged_at", { ascending: false })
    .limit(1)
    .single();

  if (!data) return null;
  const { error } = await supabase.from("meals").delete().eq("id", data.id);
  if (error) throw new Error(error.message);
  return data.id;
}

// ── Commands ──────────────────────────────────────────────────────────────────
bot.onText(/\/start/, (msg) => {
  bot.sendMessage(
    msg.chat.id,
    `👋 Welcome to your *Calorie Tracker*!\n\nHere's how to use me:\n\n📸 *Send a photo* of your food → I'll analyze and log it\n✍️ *Describe your food* in text → I'll estimate and log it\n\n*Commands:*\n/today — see today's meals & totals\n/week — see this week's summary\n/undo — delete your last logged meal\n/help — show this message`,
    { parse_mode: "Markdown" }
  );
});

bot.onText(/\/help/, (msg) => {
  bot.sendMessage(
    msg.chat.id,
    `*Commands:*\n/today — today's meals & calorie totals\n/week — this week's breakdown + AI analysis\n/undo — remove last logged meal\n/help — show this message\n\n*To log food:*\n📸 Send a photo of your meal\n✍️ Or just type what you ate (e.g. "2 scrambled eggs and toast")`,
    { parse_mode: "Markdown" }
  );
});

bot.onText(/\/today/, async (msg) => {
  const chatId = msg.chat.id;
  bot.sendChatAction(chatId, "typing");

  try {
    const meals = await getTodaysMeals(chatId);
    if (meals.length === 0) {
      return bot.sendMessage(chatId, "No meals logged today yet! Send a photo or describe what you ate.");
    }

    const totals = formatTotals(meals);
    const mealLines = meals.map((m, i) => `${i + 1}. ${m.food_name} — ${m.calories} kcal`).join("\n");

    bot.sendMessage(
      chatId,
      `📅 *Today's Food Log*\n\n${mealLines}\n\n*Daily Totals:*\n🔥 Calories: ${totals.calories} kcal\n💪 Protein: ${totals.protein}g\n🍞 Carbs: ${totals.carbs}g\n🥑 Fat: ${totals.fat}g\n🌿 Fiber: ${totals.fiber}g\n🍬 Sugar: ${totals.sugar}g`,
      { parse_mode: "Markdown" }
    );
  } catch (err) {
    console.error(err);
    bot.sendMessage(chatId, "Error fetching today's meals. Try again!");
  }
});

bot.onText(/\/week/, async (msg) => {
  const chatId = msg.chat.id;
  bot.sendChatAction(chatId, "typing");

  try {
    const endDate = today();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 6);
    const startDateStr = startDate.toISOString().split("T")[0];

    const meals = await getMealsForRange(chatId, startDateStr, endDate);
    if (meals.length === 0) {
      return bot.sendMessage(chatId, "No meals logged this week yet!");
    }

    const totals = formatTotals(meals);
    const avgCalories = Math.round(totals.calories / 7);
    const summary = await getAISummary(meals, "this week");

    bot.sendMessage(
      chatId,
      `📊 *This Week's Summary* (${startDateStr} → ${endDate})\n\n*Totals:*\n🔥 Calories: ${totals.calories} kcal (avg ${avgCalories}/day)\n💪 Protein: ${totals.protein}g\n🍞 Carbs: ${totals.carbs}g\n🥑 Fat: ${totals.fat}g\n🌿 Fiber: ${totals.fiber}g\n🍬 Sugar: ${totals.sugar}g\n📝 Meals logged: ${meals.length}\n\n*AI Analysis:*\n${summary}`,
      { parse_mode: "Markdown" }
    );
  } catch (err) {
    console.error(err);
    bot.sendMessage(chatId, "Error fetching weekly data. Try again!");
  }
});

bot.onText(/\/undo/, async (msg) => {
  const chatId = msg.chat.id;
  try {
    const deleted = await deleteLastMeal(chatId);
    if (!deleted) return bot.sendMessage(chatId, "No meals to undo!");
    bot.sendMessage(chatId, "✅ Last meal removed!");
  } catch (err) {
    console.error(err);
    bot.sendMessage(chatId, "Error removing last meal. Try again!");
  }
});

// ── Photo handler ─────────────────────────────────────────────────────────────
bot.on("photo", async (msg) => {
  const chatId = msg.chat.id;
  bot.sendChatAction(chatId, "typing");
  bot.sendMessage(chatId, "📸 Got your photo! Analyzing the food...");

  try {
    const photo = msg.photo[msg.photo.length - 1];
    const file = await bot.getFile(photo.file_id);
    const imageUrl = `https://api.telegram.org/file/bot${process.env.TELEGRAM_BOT_TOKEN}/${file.file_path}`;

    const nutrition = await analyzeFoodImage(imageUrl);
    await saveMeal(chatId, nutrition);

    bot.sendMessage(
      chatId,
      `✅ *Meal logged!*\n\n${formatMeal(nutrition)}\n\n_${nutrition.notes}_\n\nType /today to see your daily totals.`,
      { parse_mode: "Markdown" }
    );
  } catch (err) {
    console.error(err);
    bot.sendMessage(chatId, "Couldn't analyze that photo. Try a clearer shot or describe the meal in text!");
  }
});

// ── Text message handler ──────────────────────────────────────────────────────
bot.on("message", async (msg) => {
  if (msg.text && msg.text.startsWith("/")) return;
  if (msg.photo) return;
  if (!msg.text) return;

  const chatId = msg.chat.id;
  bot.sendChatAction(chatId, "typing");

  try {
    const nutrition = await analyzeFoodText(msg.text);
    await saveMeal(chatId, nutrition);

    bot.sendMessage(
      chatId,
      `✅ *Meal logged!*\n\n${formatMeal(nutrition)}\n\n_${nutrition.notes}_\n\nType /today to see your daily totals.`,
      { parse_mode: "Markdown" }
    );
  } catch (err) {
    console.error(err);
    bot.sendMessage(chatId, "Couldn't understand that. Try describing your meal more clearly, like: '2 eggs, toast, and orange juice'");
  }
});

// ── Keep Render happy with a dummy HTTP server ────────────────────────────────
const http = require("http");
http.createServer((req, res) => res.end("OK")).listen(process.env.PORT || 3000);

console.log("🥗 Calorie Tracker Bot is running with Claude AI...");
