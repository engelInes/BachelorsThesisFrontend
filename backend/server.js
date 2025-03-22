const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const multer = require("multer");
const { GridFsStorage } = require("multer-gridfs-storage");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

mongoose
  .connect(process.env.MONGODB_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => {
    console.log("Connected to MongoDB");

    const storage = new GridFsStorage({
      url: process.env.MONGODB_URI,
      file: (req, file) => {
        console.log("Uploading file:", file);
        return {
          filename: `${Date.now()}-${file.originalname}`,
          bucketName: "images",
        };
      },
    });

    const upload = multer({ storage });

    app.post("/api/upload", upload.single("image"), async (req, res) => {
      console.log("File uploaded:", req.file);
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      try {
        const image = new Image({
          filename: req.file.filename,
          originalname: req.file.originalname,
          user: req.user.id,
        });

        await image.save();

        await User.findByIdAndUpdate(
          req.user.id,
          { $push: { images: image._id } },
          { new: true }
        );

        return res.status(200).json({
          id: req.file.id,
          filename: req.file.filename,
          message: "File uploaded successfully",
        });
      } catch (error) {
        console.error("Error saving image reference:", error);
        return res.status(500).json({ error: error.message });
      }
    });

    app.get("/api/images/:filename", async (req, res) => {
      console.log("request for image:", req.params.filename);
      try {
        const bucket = new mongoose.mongo.GridFSBucket(mongoose.connection.db, {
          bucketName: "images",
        });

        const files = await mongoose.connection.db
          .collection("images.files")
          .findOne({
            filename: req.params.filename,
          });

        console.log("found files: ", files);
        if (!files) {
          return res.status(404).json({ error: "File not found" });
        }

        const downloadStream = bucket.openDownloadStreamByName(
          req.params.filename
        );
        downloadStream.pipe(res);
      } catch (error) {
        console.error("Error getting image:", error);
        res.status(500).json({ error: error.message });
      }
    });
  })
  .catch((err) => console.error("Failed to connect to MongoDB", err));

const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  accountType: { type: String, enum: ["user", "admin"], default: "user" },
  images: [{ type: mongoose.Schema.Types.ObjectId, ref: "Image" }],
  createdAt: { type: Date, default: Date.now },
});

const User = mongoose.model("User", userSchema);

const imageSchema = new mongoose.Schema({
  filename: { type: String, required: true },
  originalname: { type: String },
  user: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  createdAt: { type: Date, default: Date.now },
});

const Image = mongoose.model("Image", imageSchema);

app.post("/api/signup", async (req, res) => {
  try {
    const { username, email, password, accountType } = req.body;

    console.log("Signup request body:", req.body);

    const existingUser = await User.findOne({ $or: [{ email }, { username }] });
    if (existingUser) {
      return res.status(400).json({ error: "User already exists" });
    }

    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    const user = new User({
      username,
      email,
      password: hashedPassword,
      accountType: accountType || "user",
    });

    await user.save();

    console.log("created user: ", user);

    const token = jwt.sign(
      { id: user._id, username: user.username, accountType: user.accountType },
      process.env.JWT_SECRET,
      { expiresIn: "7d" }
    );

    res.status(201).json({
      token,
      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        accountType: user.accountType,
      },
    });
  } catch (error) {
    console.error("Signup error:", error);
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    console.log("login req:", req.body);
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ error: "Invalid credentials" });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ error: "Invalid credentials" });
    }

    console.log("User logged in:", user);

    const token = jwt.sign(
      { id: user._id, username: user.username, accountType: user.accountType },
      process.env.JWT_SECRET,
      { expiresIn: "7d" }
    );

    res.status(200).json({
      token,
      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        accountType: user.accountType,
      },
    });
  } catch (error) {
    console.log("Login error", error);
    res.status(500).json({ error: error.message });
  }
});

const auth = (req, res, next) => {
  const token = req.header("x-auth-token");

  if (!token) {
    return res.status(401).json({ error: "No token, authorization denied" });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (error) {
    res.status(401).json({ error: "Token is not valid" });
  }
};

app.get("/api/user", auth, async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select("-password");
    console.log("Authenticated user: ", user);
    res.json(user);
  } catch (error) {
    console.error("Error fetching user: ", error);
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/user/images", auth, async (req, res) => {
  try {
    const images = await Image.find({ user: req.user.id }).sort({
      createdAt: -1,
    });

    console.log("user images:", images);
    res.json(images);
  } catch (error) {
    console.log("error fetching images:", error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
