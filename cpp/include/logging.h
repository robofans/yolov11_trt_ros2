/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#ifndef TENSORRT_LOGGING_H
#define TENSORRT_LOGGING_H

#include "NvInferRuntimeCommon.h"
#include <cassert>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include "macros.h"

using Severity = nvinfer1::ILogger::Severity;

class LogStreamConsumerBuffer : public std::stringbuf
{
public:
    LogStreamConsumerBuffer(std::ostream& stream, const std::string& prefix, bool shouldLog)
        : mOutput(stream), mPrefix(prefix), mShouldLog(shouldLog) {}
    ~LogStreamConsumerBuffer() {
        if (pbase() != pptr()) { putOutput(); }
    }
    virtual int sync() { putOutput(); return 0; }
    void putOutput() {
        if (mShouldLog) {
            std::time_t timestamp = std::time(nullptr);
            tm* tm_local = std::localtime(&timestamp);
            std::cout << "[" << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/"
                      << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/"
                      << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-"
                      << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":"
                      << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":"
                      << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
            mOutput << mPrefix << str();
            str("");
            mOutput.flush();
        }
    }
    void setShouldLog(bool shouldLog) { mShouldLog = shouldLog; }
private:
    std::ostream& mOutput;
    std::string mPrefix;
    bool mShouldLog;
};

class LogStreamConsumerBase {
public:
    LogStreamConsumerBase(std::ostream& stream, const std::string& prefix, bool shouldLog)
        : mBuffer(stream, prefix, shouldLog) {}
protected:
    LogStreamConsumerBuffer mBuffer;
};

class LogStreamConsumer : protected LogStreamConsumerBase, public std::ostream {
public:
    LogStreamConsumer(Severity reportableSeverity, Severity severity)
        : LogStreamConsumerBase(severityOstream(severity), severityPrefix(severity), severity <= reportableSeverity)
        , std::ostream(&mBuffer)
        , mShouldLog(severity <= reportableSeverity)
        , mSeverity(severity) {}
    void setReportableSeverity(Severity severity) {
        mShouldLog = mSeverity <= severity;
        mBuffer.setShouldLog(mShouldLog);
    }
private:
    static std::ostream& severityOstream(Severity severity) {
        return severity >= Severity::kINFO ? std::cout : std::cerr;
    }
    static std::string severityPrefix(Severity severity) {
        switch (severity) {
        case Severity::kINTERNAL_ERROR: return "[F] ";
        case Severity::kERROR: return "[E] ";
        case Severity::kWARNING: return "[W] ";
        case Severity::kINFO: return "[I] ";
        case Severity::kVERBOSE: return "[V] ";
        default: assert(0); return "";
        }
    }
    bool mShouldLog;
    Severity mSeverity;
};

class Logger : public nvinfer1::ILogger {
public:
    Logger(Severity severity = Severity::kWARNING) : mReportableSeverity(severity) {}
    void log(Severity severity, const char* msg) TRT_NOEXCEPT override {
        LogStreamConsumer(mReportableSeverity, severity) << "[TRT] " << std::string(msg) << std::endl;
    }
    void setReportableSeverity(Severity severity) { mReportableSeverity = severity; }
    Severity getReportableSeverity() const { return mReportableSeverity; }
private:
    Severity mReportableSeverity;
};

#endif // TENSORRT_LOGGING_H