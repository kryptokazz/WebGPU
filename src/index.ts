import { mat4 } from 'gl-matrix';

// Initialize WebGPU
async function initWebGPU() {
  // Get the GPU adapter and device
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.error('WebGPU not supported on this browser.');
    return;
  }
  const device = await adapter.requestDevice();

  // Get the canvas context
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
  const context = canvas.getContext('webgpu') as unknown as GPUCanvasContext;

  if (!context) {
    console.error('WebGPU context is not available.');
    return;
  }  // Configure the context



  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: 'opaque',
  });

  // Define vertices and indices for a cube
  const cubeVertices = new Float32Array([
    // Positions        // Colors
    -1, -1, -1,         1, 0, 0, // Vertex 0
     1, -1, -1,         0, 1, 0, // Vertex 1
     1,  1, -1,         0, 0, 1, // Vertex 2
    -1,  1, -1,         1, 1, 0, // Vertex 3
    -1, -1,  1,         1, 0, 1, // Vertex 4
     1, -1,  1,         0, 1, 1, // Vertex 5
     1,  1,  1,         1, 1, 1, // Vertex 6
    -1,  1,  1,         0, 0, 0, // Vertex 7
  ]);

  const cubeIndices = new Uint16Array([
    // Front face
    0, 1, 2,  2, 3, 0,
    // Back face
    4, 5, 6,  6, 7, 4,
    // Left face
    0, 4, 7,  7, 3, 0,
    // Right face
    1, 5, 6,  6, 2, 1,
    // Top face
    3, 7, 6,  6, 2, 3,
    // Bottom face
    0, 4, 5,  5, 1, 0,
  ]);

  // Create vertex buffer
  const vertexBuffer = device.createBuffer({
    size: cubeVertices.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(vertexBuffer.getMappedRange()).set(cubeVertices);
  vertexBuffer.unmap();

  // Create index buffer
  const indexBuffer = device.createBuffer({
    size: cubeIndices.byteLength,
    usage: GPUBufferUsage.INDEX,
    mappedAtCreation: true,
  });
  new Uint16Array(indexBuffer.getMappedRange()).set(cubeIndices);
  indexBuffer.unmap();

  // Create uniform buffer for the MVP matrix
  const uniformBufferSize = 64; // 4x4 matrix
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create shader modules
  const vertexShaderModule = device.createShaderModule({
    code: `
      struct Uniforms {
        mvpMatrix : mat4x4<f32>;
      };
      @binding(0) @group(0) var<uniform> uniforms : Uniforms;

      struct VertexOutput {
        @builtin(position) Position : vec4<f32>;
        @location(0) vColor : vec3<f32>;
      };

      @vertex
      fn main(
        @location(0) position : vec3<f32>,
        @location(1) color : vec3<f32>
      ) -> VertexOutput {
        var output : VertexOutput;
        output.Position = uniforms.mvpMatrix * vec4<f32>(position, 1.0);
        output.vColor = color;
        return output;
      }
    `,
  });

  const fragmentShaderModule = device.createShaderModule({
    code: `
      @fragment
      fn main(@location(0) vColor : vec3<f32>) -> @location(0) vec4<f32> {
        return vec4<f32>(vColor, 1.0);
      }
    `,
  });

  // Create pipeline layout and bind group
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: 'uniform' },
    }],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{
      binding: 0,
      resource: { buffer: uniformBuffer },
    }],
  });

  // Create render pipeline
  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: vertexShaderModule,
      entryPoint: 'main',
      buffers: [{
        arrayStride: 6 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' }, // Position
          { shaderLocation: 1, offset: 3 * 4, format: 'float32x3' }, // Color
        ],
      }],
    },
    fragment: {
      module: fragmentShaderModule,
      entryPoint: 'main',
      targets: [{ format }],
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'back',
    },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
  });

  // Create depth texture
  const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Animation loop
  function frame() {
    // Update the uniform buffer with the latest MVP matrix
    const aspect = canvas.width / canvas.height;
    const projectionMatrix = mat4.perspective(mat4.create(), Math.PI / 4, aspect, 0.1, 100);
    const viewMatrix = mat4.lookAt(mat4.create(), [4, 3, 10], [0, 0, 0], [0, 1, 0]);
    const modelMatrix = mat4.rotateY(mat4.create(), mat4.create(), performance.now() / 1000);
    const mvpMatrix = mat4.multiply(mat4.create(), projectionMatrix, mat4.multiply(mat4.create(), viewMatrix, modelMatrix));

    device.queue.writeBuffer(uniformBuffer, 0, mvpMatrix as Float32Array);

    // Begin render pass
    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0.2, g: 0.2, b: 0.2, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setIndexBuffer(indexBuffer, 'uint16');
    renderPass.drawIndexed(cubeIndices.length);
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

initWebGPU();

