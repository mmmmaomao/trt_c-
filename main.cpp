#include <iostream>
#include <fstream>
#include <cassert>

#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>


class Logger : public nvinfer1::ILogger
{
    //void log(Severity severity, const char* msg) override
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;


int main() {
    // 实例化builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    // 加载onnx文件
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    const char* onnx_filename = "/home/w/PythonProjects/TensorRT/gan_model.onnx";
    parser->parseFromFile(onnx_filename, static_cast<int>(Logger::Severity::kWARNING));
    for (int i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    std::cout << "successfully load the onnx model" << std::endl;

    // 创建引擎
    unsigned int maxBatchSize = 1;
    builder->setMaxBatchSize(maxBatchSize);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 30);
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // 序列化
    nvinfer1::IHostMemory* serializedModel = engine->serialize();
    std::ofstream serialize_output_stream("/home/w/PythonProjects/TensorRT/gan_engine.trt", std::ios_base::out | std::ios_base::binary);
    if(serialize_output_stream.good()){
        std::cout<<serializedModel->size()<<std::endl;
        serialize_output_stream.write((char*)serializedModel->data(), serializedModel->size());
        serialize_output_stream.close();
    }
    else{
        exit(-1);
    }

	
    delete parser;
    delete network;
    delete config;
    delete builder;
    delete engine;
    
	
    // 反序列化
    //engine文件参数读取后保存到该数组中
    std::vector<char> trtModelStream_;       
    size_t size{0};
    std::string cached_path = "/home/w/PythonProjects/TensorRT/gan_engine.trt";
    std::ifstream trtModelFile(cached_path, std::ios_base::in | std::ios_base::binary);

    if (trtModelFile.good()) {
        trtModelFile.seekg(0, std::ios::end);
        size = trtModelFile.tellg();
        std::cout<<size<<std::endl;
        trtModelFile.seekg(0, std::ios::beg);
        trtModelStream_.resize(size);
        trtModelFile.read(trtModelStream_.data(), size);
    }
    else
    {
        exit(-1);
    }
    

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    nvinfer1::ICudaEngine* re_engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, NULL);
    assert(re_engine != nullptr);
    std::cout << "successfully deserial the cuda engine" << std::endl;
    trtModelFile.close();

    //创建context
    nvinfer1::IExecutionContext* context = re_engine->createExecutionContext();
    assert(context != nullptr);

    // 输入处理
    // input需要从仿真器中获取
    // eps是正态分布的向量
    std::vector<float> input(350,1);  
    std::vector<float> eps(128,1);

    //创建buffers 指向输入输出流
    void* buffers[5];
    int32_t inputIndex = re_engine->getBindingIndex("input");
    int32_t epsIndex = re_engine->getBindingIndex("eps");
    int32_t outputIndex = re_engine->getBindingIndex("output");
    int32_t muIndex = re_engine->getBindingIndex("mu");
    int32_t valIndex = re_engine->getBindingIndex("val");

    // 分配buffers空间
    auto err = cudaMalloc(&buffers[inputIndex], 350 * sizeof(float));
    assert(err==0);
    err = cudaMalloc(&buffers[epsIndex], 128 * sizeof(float));
    assert(err==0);
    err = cudaMalloc(&buffers[outputIndex], 140 * sizeof(float));
    assert(err==0);
    err = cudaMalloc(&buffers[muIndex], 128 * sizeof(float));
    assert(err==0);
    err = cudaMalloc(&buffers[valIndex], 128 * sizeof(float));
    assert(err==0);

    //创建cuda流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //复制输入数据到GPU
    err = cudaMemcpy(buffers[inputIndex], input.data(), 350* sizeof(float), cudaMemcpyHostToDevice);
    assert(err==0);
    err = cudaMemcpy(buffers[epsIndex], input.data(), 128* sizeof(float), cudaMemcpyHostToDevice);
    assert(err==0);

    //执行推理
    clock_t start_time=clock();
    context->enqueue(1, buffers, stream, nullptr);
    clock_t end_time=clock();

    std::cout<< "Running time is: "<<static_cast<double>(end_time-start_time)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;
    //将GPU数据拷贝回CPU
    float opt[140], mu[128], val[128];
    err = cudaMemcpy(opt, buffers[outputIndex], 140* sizeof(float), cudaMemcpyDeviceToHost);
    assert(err==0);
    err = cudaMemcpy(mu, buffers[muIndex], 128* sizeof(float), cudaMemcpyDeviceToHost);
    assert(err==0);
    err = cudaMemcpy(val, buffers[valIndex], 128* sizeof(float), cudaMemcpyDeviceToHost);
    assert(err==0);
    
    // 处理输入，opt需要reshape一下
    for(auto value:opt)
        std::cout<<value<<std::endl;

    //释放资源
    cudaStreamDestroy(stream);
    context->destroy();
    re_engine->destroy();
    runtime->destroy();
    cudaFree(buffers[inputIndex]);
    
    return 0;
}

