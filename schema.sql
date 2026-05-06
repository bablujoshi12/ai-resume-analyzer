-- AI Resume Analyzer - MySQL Schema
-- This file creates the database and required tables.

CREATE DATABASE IF NOT EXISTS cv;
USE cv;

-- Admin login table
CREATE TABLE IF NOT EXISTS admin_users (
  id INT NOT NULL AUTO_INCREMENT,
  username VARCHAR(100) NOT NULL UNIQUE,
  password_hash VARCHAR(128) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
);

-- User accounts table (for user login)
CREATE TABLE IF NOT EXISTS user_accounts (
  id INT NOT NULL AUTO_INCREMENT,
  name VARCHAR(120) NOT NULL,
  email VARCHAR(180) NOT NULL UNIQUE,
  password_hash VARCHAR(128) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
);

-- Main analysis records
CREATE TABLE IF NOT EXISTS user_data (
  ID INT NOT NULL AUTO_INCREMENT,
  user_id INT NULL,
  sec_token VARCHAR(20) NOT NULL,
  ip_add VARCHAR(50) NULL,
  host_name VARCHAR(50) NULL,
  dev_user VARCHAR(50) NULL,
  os_name_ver VARCHAR(50) NULL,
  latlong VARCHAR(50) NULL,
  city VARCHAR(50) NULL,
  state VARCHAR(50) NULL,
  country VARCHAR(50) NULL,
  act_name VARCHAR(50) NOT NULL,
  act_mail VARCHAR(50) NOT NULL,
  act_mob VARCHAR(20) NOT NULL,
  Name VARCHAR(500) NOT NULL,
  Email_ID VARCHAR(500) NOT NULL,
  resume_score VARCHAR(8) NOT NULL,
  Timestamp VARCHAR(50) NOT NULL,
  Page_no VARCHAR(5) NOT NULL,
  Predicted_Field BLOB NOT NULL,
  User_level BLOB NOT NULL,
  Actual_skills BLOB NOT NULL,
  Recommended_skills BLOB NOT NULL,
  Recommended_courses BLOB NOT NULL,
  pdf_name VARCHAR(50) NOT NULL,
  PRIMARY KEY (ID),
  INDEX idx_user_data_user_id (user_id)
);

-- Feedback table
CREATE TABLE IF NOT EXISTS user_feedback (
  ID INT NOT NULL AUTO_INCREMENT,
  feed_name VARCHAR(50) NOT NULL,
  feed_email VARCHAR(50) NOT NULL,
  feed_score VARCHAR(5) NOT NULL,
  comments VARCHAR(100) NULL,
  Timestamp VARCHAR(50) NOT NULL,
  PRIMARY KEY (ID)
);

