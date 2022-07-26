use std::{f32::consts::PI, iter};

use lazy_static::lazy_static;
use rand::{prelude::ThreadRng, Rng};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod texture;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

lazy_static! {
    static ref DATA: (&'static [(f32, f32, f32)], [&'static [u16]; 4]) = {
        let mut vertices = vec![];
        let mut indices = [vec![], vec![], vec![], vec![]];
        let mut rng = rand::thread_rng();
        fn add_vertex(vertices: &mut Vec<(f32, f32, f32)>, a: f32, b: f32, c: f32) -> u16 {
            let r = vertices.len();
            vertices.push((a, b, c));
            r as u16
        }
        fn add_face(indices: &mut [Vec<u16>; 4], a: u16, b: u16, c: u16, tx: usize) {
            indices[tx % 4].extend([a, b, c]);
        }
        fn add_random_face(
            indices: &mut [Vec<u16>; 4],
            vertices: &mut Vec<(f32, f32, f32)>,
            a: u16,
            b: u16,
            c: u16,
            rng: &mut ThreadRng,
        ) {
            let choice = rng.gen_range(0i32..10);
            match choice {
                0..=1 => {
                    let d = {
                        let r = rng.gen_range(0.2..0.8);
                        let (x, y, z) = vertices[a as usize];
                        let (n, m, p) = vertices[b as usize];
                        let q = 1. - r;
                        add_vertex(vertices, x * r + n * q, y * r + m * q, z * r + p * q)
                    };
                    add_face(indices, a, d, c, rng.gen());
                    add_random_face(indices, vertices, d, b, c, rng);
                }
                2..=7 => {
                    let d = {
                        let r = rng.gen_range(0.2..0.8);
                        let (x, y, z) = vertices[a as usize];
                        let (n, m, p) = vertices[b as usize];
                        let q = 1. - r;
                        add_vertex(vertices, x * r + n * q, y * r + m * q, z * r + p * q)
                    };
                    let e = {
                        let r = rng.gen_range(0.2..0.8);
                        let (x, y, z) = vertices[a as usize];
                        let (n, m, p) = vertices[c as usize];
                        let q = 1. - r;
                        add_vertex(vertices, x * r + n * q, y * r + m * q, z * r + p * q)
                    };
                    add_face(indices, a, b, c, rng.gen());
                    add_random_face(indices, vertices, d, e, a, rng);
                }
                8..=9 => add_face(indices, a, b, c, rng.gen()),
                _ => unreachable!(),
            }
        }
        const N: usize = 10;
        const N_F32: f32 = N as f32;
        let down_id = add_vertex(&mut vertices, 0., -1., 0.);
        let (big_circle, small_circle): (Vec<_>, Vec<_>) = (0..N)
            .map(|i| {
                let i_f32 = i as f32;
                let x = f32::sin(i_f32 / N_F32 * 2. * PI);
                let y = f32::cos(i_f32 / N_F32 * 2. * PI);
                let a = add_vertex(&mut vertices, x, 0., y);
                let b = add_vertex(&mut vertices, x / 2., 0.5, y / 2.);
                (a, b)
            })
            .unzip();
        let top_id = add_vertex(&mut vertices, 0., 0.5, 0.);
        for (&x, &y) in big_circle.iter().zip(big_circle.iter().cycle().skip(1)) {
            add_random_face(&mut indices, &mut vertices, down_id, x, y, &mut rng);
        }
        for (&x, &y) in small_circle.iter().zip(small_circle.iter().cycle().skip(1)) {
            add_face(&mut indices, x, top_id, y, 0);
        }
        let big_small_zip = big_circle.iter().zip(small_circle.iter());
        for ((&x, &y), (&nx, &ny)) in big_small_zip.clone().zip(big_small_zip.cycle().skip(1)) {
            add_random_face(&mut indices, &mut vertices, x, y, nx, &mut rng);
            add_random_face(&mut indices, &mut vertices, nx, y, ny, &mut rng);
        }
        let indices = indices.map(|x| -> &'static [u16] { Box::leak(x.into_boxed_slice()) });
        (Box::leak(vertices.into_boxed_slice()), indices)
    };
    static ref VERTICES: &'static [(f32, f32, f32)] = DATA.0;
    static ref INDICES_TX1: &'static [u16] = DATA.1[0];
    static ref INDICES_TX2: &'static [u16] = DATA.1[1];
    static ref INDICES_TX3: &'static [u16] = DATA.1[2];
    static ref INDICES_TX4: &'static [u16] = DATA.1[3];
    static ref TX1_CNT: u64 = INDICES_TX1.len() as u64;
    static ref TX2_CNT: u64 = INDICES_TX2.len() as u64;
    static ref TX3_CNT: u64 = INDICES_TX3.len() as u64;
    static ref TX4_CNT: u64 = INDICES_TX4.len() as u64;
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        proj * view
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = (OPENGL_TO_WGPU_MATRIX * camera.build_view_projection_matrix()).into();
    }
}

struct CameraController {
    speed: f32,
    zoom_offset: f32,
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_clicked: bool,
    prev_cursor: (f64, f64),
    current_cursor: (f64, f64),
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            zoom_offset: 0.,
            is_up_pressed: false,
            is_down_pressed: false,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_clicked: false,
            prev_cursor: (0., 0.),
            current_cursor: (0., 0.),
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                if *dbg!(button) == MouseButton::Left {
                    self.is_clicked = *dbg!(state) == ElementState::Pressed;
                    self.prev_cursor = self.current_cursor;
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.current_cursor = (position.x, position.y);
                if !self.is_clicked {
                    self.prev_cursor = self.current_cursor;
                    return false;
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                match delta {
                    MouseScrollDelta::LineDelta(_, x) => self.zoom_offset += x,
                    _ => unreachable!(),
                }
                true
            }
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::Space => {
                        self.is_up_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::LShift => {
                        self.is_down_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn update_camera(&mut self, camera: &mut Camera) {
        use cgmath::InnerSpace;
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        let right = forward_norm.cross(camera.up);
        let rotate_vec = (self.prev_cursor.0 - self.current_cursor.0) as f32 * right
            - (self.prev_cursor.1 - self.current_cursor.1) as f32 * camera.up;
        camera.eye = camera.target - (forward + rotate_vec * self.speed).normalize() * forward_mag;
        self.prev_cursor = self.current_cursor;

        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();

        camera.eye += forward_norm * self.zoom_offset;
        self.zoom_offset = 0.;

        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye -= cgmath::vec3(0., 0.3, 0.);
        }
        if self.is_backward_pressed {
            camera.eye += cgmath::vec3(0., 0.3, 0.);
        }

        if self.is_right_pressed {
            camera.eye -= cgmath::vec3(0.3, 0., 0.);
        }
        if self.is_left_pressed {
            camera.eye += cgmath::vec3(0.3, 0., 0.);
        }
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    diffuse_bind_group: wgpu::BindGroup,
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    diffuse_bind_group2: wgpu::BindGroup,
    diffuse_bind_group3: wgpu::BindGroup,
    diffuse_bind_group4: wgpu::BindGroup,
}

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),

                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                },
                None,
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let diffuse_texture = include_texture!(device, queue, "texture1.png");

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let diffuse_texture2 = include_texture!(device, queue, "texture2.png");

        let texture2_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture2_bind_group_layout"),
            });

        let diffuse_bind_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture2_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture2.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture2.sampler),
                },
            ],
            label: Some("diffuse_bind_group2"),
        });

        let diffuse_texture3 = include_texture!(device, queue, "texture3.png");

        let texture3_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture2_bind_group_layout"),
            });

        let diffuse_bind_group3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture3_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture3.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture3.sampler),
                },
            ],
            label: Some("diffuse_bind_group3"),
        });

        let diffuse_texture4 = include_texture!(device, queue, "texture4.png");

        let texture4_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture4_bind_group_layout"),
            });

        let diffuse_bind_group4 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture4_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture4.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture2.sampler),
                },
            ],
            label: Some("diffuse_bind_group4"),
        });

        let camera = Camera {
            eye: (0.0, 1.0, 2.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 15.0,
            znear: 0.1,
            zfar: 100.0,
        };
        let camera_controller = CameraController::new(0.1);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),

                polygon_mode: wgpu::PolygonMode::Fill,

                unclipped_depth: false,

                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let mut rng = rand::thread_rng();

        let vertices = Box::leak(
            VERTICES
                .iter()
                .map(|&(a, b, c)| Vertex {
                    position: [a, b, c],
                    tex_coords: [rng.gen_range(0.0..1.), rng.gen_range(0.0..1.)],
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );

        let indices = Box::leak(
            [*INDICES_TX1, *INDICES_TX2, *INDICES_TX3, *INDICES_TX4]
                .concat()
                .into_boxed_slice(),
        );

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            diffuse_bind_group,
            diffuse_bind_group2,
            diffuse_bind_group3,
            diffuse_bind_group4,
            camera,
            camera_controller,
            camera_buffer,
            camera_bind_group,
            camera_uniform,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            self.camera.aspect = self.config.width as f32 / self.config.height as f32;
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.,
                            g: 0.,
                            b: 0.,
                            a: 1.,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                self.index_buffer.slice(0..INDICES_TX1.len() as u64 * 2),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.draw_indexed(0..INDICES_TX1.len() as u32, 0, 0..1);
            render_pass.set_index_buffer(
                self.index_buffer.slice(
                    INDICES_TX1.len() as u64 * 2
                        ..(INDICES_TX1.len() + INDICES_TX2.len()) as u64 * 2,
                ),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.set_bind_group(0, &self.diffuse_bind_group2, &[]);
            render_pass.draw_indexed(0..INDICES_TX2.len() as u32, 0, 0..1);
            render_pass.set_index_buffer(
                self.index_buffer.slice(
                    (INDICES_TX1.len() + INDICES_TX2.len()) as u64 * 2
                        ..(INDICES_TX1.len() + INDICES_TX2.len() + INDICES_TX3.len()) as u64 * 2,
                ),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.set_bind_group(0, &self.diffuse_bind_group3, &[]);
            render_pass.draw_indexed(0..INDICES_TX3.len() as u32, 0, 0..1);
            render_pass.set_index_buffer(
                self.index_buffer.slice(
                    (INDICES_TX1.len() + INDICES_TX2.len() + INDICES_TX3.len()) as u64 * 2
                        ..(INDICES_TX1.len()
                            + INDICES_TX2.len()
                            + INDICES_TX3.len()
                            + INDICES_TX4.len()) as u64
                            * 2,
                ),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.set_bind_group(0, &self.diffuse_bind_group4, &[]);
            render_pass.draw_indexed(0..INDICES_TX4.len() as u32, 0, 0..1);
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = State::new(&window).await;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            state.update();
            match state.render() {
                Ok(_) => {}

                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    state.resize(state.size)
                }

                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,

                Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
            }
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    });
}
