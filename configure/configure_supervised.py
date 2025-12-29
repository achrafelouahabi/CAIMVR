def get_default_config(data_name):
    if data_name in ['Caltech101-7']:
        return dict(
            seed=5,
            view=2,
            training=dict(
                pretrain_epochs=150,
                batch_size=256,
                epoch=500,
                alpha=10,
                lambda2=0.1,
                lambda1=0.11,
                lr=1.0e-4,
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],
                arch2=[512, 1024, 1024, 1024, 128],
                activations1='gelu',
                activations2='gelu',
                batchnorm=True,
                heads=16
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                activations1='gelu',
                activations2='gelu',
                heads=8
            )
 
        )

    elif data_name in ['Scene_15']:
        """The default configs."""
        return dict(
            seed=7,
            view=2,
            k=15,
            training=dict(
                lr=1.0e-3,
                pretrain_epochs=50,
                batch_size=1024,
                epoch=500,
                alpha=10,
                lambda2=0.1,
                lambda1=0.11,
            
            ),
            Autoencoder=dict(
                arch1=[40, 1024, 1024, 1024, 128],
                arch2=[59, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
                heads=16
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                activations1='relu',
                activations2='relu',
                heads=4
            ) )


    elif data_name in ['hand']:
        """The default configs."""
        return dict(
            seed=5,
            view=2,
            k=5,
            training=dict(
                lr=1.0e-4,
                pretrain_epochs=150,
                batch_size=256,
                epoch=200,
                alpha=10,
                lambda2=0.1,
                lambda1=0.11,
            
            ),
            Autoencoder=dict(
                arch1=[240, 1024, 1024, 1024, 128],
                arch2=[216, 1024, 1024, 1024, 128],
                activations1='gelu',
                activations2='gelu',
                batchnorm=True,
                heads=16
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                activations1='gelu',
                activations2='gelu',
                heads=8
            ) )
    elif data_name in ['NoisyMNIST']:
        """The default configs."""
        return dict(
            seed=1,
            view=2,
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 40],
                arch2=[784, 1024, 1024, 1024, 40],
                activations1='gelu',
                activations2='gelu',
                batchnorm=True,
                heads=12
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                activations1='gelu',
                activations2='gelu',
                heads=14
            ),
            training=dict(
                lr=1.0e-4,
                pretrain_epochs=0,
                batch_size=256,
                epoch=500,
                alpha=10,
                lambda1=0.11,
                lambda2=0.1,
            ),
        )

    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            seed=2,
            view=2,
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 40],
                arch2=[40, 1024, 1024, 1024, 40],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
                heads=6
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                activations1='relu',
                activations2='relu',
                heads=16
            ),
            training=dict(
                lr=5.0e-4,
                pretrain_epochs=0,
                batch_size=256,
                epoch=500,
                epoch_3=500,
                alpha=10,
                lambda2=0.1,
                lambda1=0.11,
            ),
        )
    elif data_name in ['DHA']:
        """The default configs."""
        return dict(
            missing_rate=0,
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                activations1='gelu',
                activations2='gelu',
                heads=6
            ),
            Autoencoder=dict(
                arch1=[6144, 2048, 512, 64],
                arch2=[110, 1024, 512, 64],
                activations1='gelu',
                activations2='gelu',
                batchnorm=True,
                heads=6
            ),
            training=dict(
                lr=5.0e-4,
                pretrain_epochs=150,
                batch_size=100,
                epoch=2000,
                alpha=10,
                lambda2=0.1,
                lambda1=0.11,
            ),
            seed=26,
        )

    elif data_name in ['UWA30']:
        """The default configs."""
        return dict(
            missing_rate=0,
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                activations1='gelu',
                activations2='gelu',
                heads=4
            ),
            Autoencoder=dict(
                arch1=[6144, 2048, 512, 128],
                arch2=[110, 1024, 512, 128],
                activations1='gelu',
                activations2='gelu',
                batchnorm=True,
                heads=16
            ),
            training=dict(
                lr=4.0e-4,
                pretrain_epochs=150,
                batch_size=200,
                epoch=1000,
                alpha=10,
                lambda2=0.1,
                lambda1=0.11,
            ),
            seed=6,
        )





    else:
        raise Exception('Undefined data_name')
