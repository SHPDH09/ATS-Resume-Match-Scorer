const nodemailer = require("nodemailer");
const express = require("express");
const cors = require("cors");
const app = express();
app.use(cors());
app.use(express.json());

const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: "rk331159@gmail.com@gmail.com",
    pass: "your_app_password"
  }
});

app.post("/send", (req, res) => {
  const { to, subject, text } = req.body;

  const mailOptions = {
    from: "your_email@gmail.com",
    to,
    subject,
    text
  };

  transporter.sendMail(mailOptions, (error, info) => {
    if (error) return res.status(500).send(error.toString());
    res.status(200).send("✅ Email sent: " + info.response);
  });
});

app.listen(3000, () => console.log("Server running on port 3000"));
